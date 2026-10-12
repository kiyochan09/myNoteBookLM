"""Read-only preflight for possible duplicate document imports."""
import difflib
import json
import re
import unicodedata


def basename(name):
    return str(name or '').replace('\\', '/').rsplit('/', 1)[-1]


def normalized(name):
    name = re.sub(r'\.(docx|odt|pdf)$', '', basename(name), flags=re.I)
    name = unicodedata.normalize('NFKC', name).casefold()
    return ''.join(c for c in name if c.isalnum())


def find_duplicates(files, documents):
    results = []
    seen = []
    for index, file in enumerate(files):
        name = basename(file.get('name'))
        key = normalized(name)
        digest = file.get('sha256') or ''
        candidates = []
        for doc in documents:
            reasons = []
            source = basename(doc.get('source_filename'))
            keys = [normalized(source), normalized(doc.get('title'))]
            if source and unicodedata.normalize('NFKC', source).casefold() == unicodedata.normalize('NFKC', name).casefold():
                reasons.append('元ファイル名が一致')
            elif key and key in keys:
                reasons.append('文書名が一致（空白・記号・拡張子の違いを除く）')
            elif len(key) >= 8 and any(len(k)>=8 and difflib.SequenceMatcher(None,key,k).ratio()>=.94 for k in keys):
                reasons.append('文書名がよく似ています')
            metadata = doc.get('doc_metadata_json') or {}
            if isinstance(metadata, str):
                try: metadata = json.loads(metadata)
                except (ValueError,TypeError): metadata = {}
            if digest and isinstance(metadata,dict) and metadata.get('import_sha256') == digest:
                reasons.append('ファイル内容が一致')
            if reasons:
                candidates.append({'document_id':doc['id'], 'title':doc['title'], 'source_filename':source, 'reasons':reasons})
        for old_index, old_name, old_key, old_hash in seen:
            if (key and key == old_key) or (digest and digest == old_hash):
                candidates.append({'selected_index':old_index,'title':old_name,'reasons':['今回選択したファイル間で重複の可能性']})
        if candidates: results.append({'index':index,'filename':name,'candidates':candidates})
        seen.append((index,name,key,digest))
    return results
