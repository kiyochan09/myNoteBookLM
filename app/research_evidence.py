"""Local query variants and exact, contextual evidence; independent of the LLM."""
from hashlib import sha256
from itertools import islice
import math
import re
from collections import Counter, defaultdict

INTENT_WORDS = {'活動', '利用', '役割', '特徴', '影響', '理由', '背景', '経緯', '方法', '詳', '具体的'}


def text_language(text):
    # Script-based retrieval lanes, not a claim of authoritative language metadata.
    japanese=len(re.findall(r'[一-龯ぁ-ゖァ-ヴ]',text))
    latin=len(re.findall(r'[A-Za-z]',text))
    return 'en' if latin >= 20 and latin > japanese*4 else 'ja'


def literal_match(term):
    return 'text_content : "' + term.replace('"', '""') + '"'


def lexical_groups(question, terms, count, lookup):
    """Learn compound variants from scoped original text, never from an answer."""
    quoted = set(re.findall(r'[「『"]([^」』"\n]+)[」』"]', question))
    groups = []
    substantive = [t for t in terms if t not in INTENT_WORDS]
    for term in terms:
        records, frequency = lookup(term)
        weight = math.log(1 + (count + 1) / (frequency + 1))
        if substantive and term in INTENT_WORDS:
            weight *= .12
        aliases = []
        if term not in quoted and re.fullmatch(r'[ァ-ヴー]{6,}', term):
            variants = {part for length in range(3, len(term) - 1)
                        for part in (term[:length], term[-length:])}
            for variant in sorted(variants, key=lambda t: (not term.endswith(t), -len(t), t))[:16]:
                samples, df = lookup(variant)
                if not samples or (count >= 8 and df > count * .7):
                    continue
                # A substring must occur as a standalone katakana word in
                # the selected corpus, rather than as an arbitrary n-gram.
                boundary = re.compile(r'(?<![ァ-ヴー])' + re.escape(variant) + r'(?![ァ-ヴー])')
                if not any(boundary.search(r['text_content'] or '') for r in samples):
                    continue
                if any(variant in existing for existing in aliases):
                    continue
                aliases.append(variant)
                if len(aliases) == 1:
                    break
        groups.append({'term': term, 'aliases': aliases, 'weight': weight})
    return groups


def relevance(text, groups):
    value = text.lower()
    score = 0.; coverage = 0; found = []
    for group in groups:
        matches = [t for t in [group['term'], *group['aliases']] if t.lower() in value]
        if matches:
            strength = 1. if group['term'] in matches else .85
            score += group['weight'] * strength
            if group['term'] not in INTENT_WORDS:
                coverage += 1
            found.extend(matches)
    return score, coverage, list(dict.fromkeys(found))


def evidence_windows(text, groups, size=1000):
    """Keep distinct hits in a long OCR block; prefer windows covering the relationship."""
    lower=text.lower();positions=[]
    substantive=any(g['term'] not in INTENT_WORDS for g in groups)
    for group in groups:
        if substantive and group['term'] in INTENT_WORDS:continue
        for term in [group['term'],*group['aliases']]:
            positions.extend(match.start() for match in islice(re.finditer(re.escape(term.lower()),lower),64))
    if not positions:return []
    starts={max(0,p-240) for p in positions}
    # A hit near the end of a window should not displace the other query entity.
    starts.update(max(0,p-size+120) for p in positions)
    ranked=sorted(starts,key=lambda start:(-relevance(text[start:start+size],groups)[0],start))
    chosen=[]
    for start in ranked:
        end=min(len(text),start+size)
        if any(max(0,min(end,b)-max(start,a))>=.5*min(end-start,b-a) for a,b in chosen):continue
        chosen.append((start,end))
        if len(chosen)==3:break
    return chosen


def related_terms(groups, rows, count, lookup):
    """Find recurring topic vocabulary beside the main subject in scoped originals."""
    if not groups:return []
    primary=max(groups,key=lambda g:g['weight'])
    subjects=[primary['term'],*primary['aliases']]
    frequencies=Counter();documents=defaultdict(set);occurrences=defaultdict(set)
    existing={t for g in groups for t in [g['term'],*g['aliases']]}
    for row_index,row in enumerate(rows):
        text=row['text_content'] or '';lower=text.lower();near=[]
        for subject in subjects:
            position=lower.find(subject.lower())
            if position>=0:near.append(text[max(0,position-250):position+len(subject)+250])
        tokens=set(re.findall(r'[ァ-ヴー]{3,24}|[A-Z][A-Z0-9]{1,11}|[一-龯々]{3,12}', '\n'.join(near)))
        for token in tokens-existing-INTENT_WORDS:
            if set(token)<=set('一二三四五六七八九十百千万億兆年月日'):continue
            frequencies[token]+=1;documents[token].add(row['document_id'])
            occurrences[token].add(row_index)
    proposed=[t for t in frequencies if frequencies[t]>=2 and len(documents[t])>=2]
    proposed.sort(key=lambda t:(-len(documents[t]),-frequencies[t],t))
    scored=[]
    for token in proposed[:64]:
        samples,df=lookup(token)
        if not samples or (count>=8 and df>count*.5):continue
        score=math.log1p(frequencies[token])*math.log1p((count+1)/(df+1))
        scored.append((score,token))
    selected=[]
    while scored and len(selected)<3:
        def diverse(item):
            score,token=item
            redundancy=max((len(occurrences[token]&occurrences[other]) /
                max(1,len(occurrences[token]|occurrences[other])) for other in selected),default=0)
            return score/(1+3*redundancy)
        chosen=max(scored,key=diverse);scored.remove(chosen);selected.append(chosen[1])
    return selected


class ContextExpander:
    def __init__(self, conn):
        self.conn = conn
        self.cache = {}

    def expand(self, citation):
        doc_id = citation['document_id']
        if doc_id not in self.cache:
            blocks = [dict(r) for r in self.conn.execute('''SELECT id AS block_id,text_content,
                page_number,section_id,reading_order FROM content_blocks WHERE document_id=?
                ORDER BY reading_order,id''', (doc_id,))]
            self.cache[doc_id] = (blocks, {b['block_id']: i for i, b in enumerate(blocks)})
        blocks, indices = self.cache[doc_id]
        original = citation.get('segments') or [citation]
        if any(s['block_id'] not in indices for s in original):
            return citation
        if any(sha256((blocks[indices[s['block_id']]]['text_content'] or '').encode()).hexdigest()
               != s['content_hash'] for s in original):
            return citation  # Keep the stale hash so the freshness check rejects it.
        first, last = indices[original[0]['block_id']], indices[original[-1]['block_id']]
        key = (blocks[first]['page_number'], blocks[first]['section_id'])
        if any((b['page_number'], b['section_id']) != key for b in blocks[first:last+1]):
            return citation
        left, right = first, last
        length = original[0]['start']
        while left > 0 and first - left < 12 and length < 650:
            previous = blocks[left-1]
            if (previous['page_number'], previous['section_id']) != key:
                break
            left -= 1; length += len(previous['text_content'] or '') + 1
        length = len(blocks[last]['text_content'] or '') - original[-1]['end']
        while right+1 < len(blocks) and right-last < 12 and length < 1100:
            following = blocks[right+1]
            if (following['page_number'], following['section_id']) != key:
                break
            right += 1; length += len(following['text_content'] or '') + 1
        text = ''; spans = []; center_start = center_end = 0
        for index in range(left, right+1):
            block = blocks[index]
            if text: text += '\n'
            start = len(text); text += block['text_content'] or ''
            spans.append((start, len(text), block))
            if index == first: center_start = start + original[0]['start']
            if index == last: center_end = start + original[-1]['end']
        start = max(0, center_start - 650)
        end = min(len(text), max(center_end, center_start + 1200) + 1100, start + 2400)
        # Prefer full sentence/paragraph boundaries without moving past the hit.
        if start:
            boundary = re.search(r'[。！？\n]', text[start:min(center_start, start+160)])
            if boundary: start += boundary.end()
        if end < len(text):
            endings = list(re.finditer(r'[。！？\n]', text[max(center_end, end-160):end]))
            if endings: end = max(center_end, end-160) + endings[-1].end()
        segments = []
        for a, b, block in spans:
            # Empty OCR blocks contain no quoted characters or valid range.
            if b > a and b > start and a < end:
                segments.append({'block_id': block['block_id'],
                    'content_hash': sha256((block['text_content'] or '').encode()).hexdigest(),
                    'start': max(start, a)-a, 'end': min(end, b)-a})
        if not segments: return citation
        leading = segments[0]
        has_prefix=start>0 or (left>0 and (blocks[left-1]['page_number'],blocks[left-1]['section_id'])==key)
        has_suffix=end<len(text) or (right+1<len(blocks) and (blocks[right+1]['page_number'],blocks[right+1]['section_id'])==key)
        return {**citation, **leading, 'segments': segments, 'quote': text[start:end],
            'reading_order': blocks[indices[leading['block_id']]]['reading_order'],
            'truncated': has_prefix or has_suffix, 'has_prefix': has_prefix,
            'has_suffix': has_suffix, 'context_expanded': True}
