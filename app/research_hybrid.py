"""Combine independently scoped keyword and vector candidates, keeping exact quotations."""
from contextlib import closing
from hashlib import sha256

from app.research_index import resolve_documents


def ranges(citation):
    return citation.get('segments') or [citation]


def overlaps(one,two,ratio=0):
    if one['document_id']!=two['document_id']:return False
    total=0
    for a in ranges(one):
        for b in ranges(two):
            if a['block_id']==b['block_id']:total+=max(0,min(a['end'],b['end'])-max(a['start'],b['start']))
    return total>0 and total/max(1,min(len(one['quote']),len(two['quote'])))>=ratio


class HybridEvidenceRetriever:
    def __init__(self,keyword,index,db_service):
        self.keyword=keyword;self.index=index;self.db_service=db_service

    def fresh(self,candidates,tag_ids):
        if not candidates:return []
        with closing(self.db_service.get_connection()) as conn:
            conn.execute('PRAGMA query_only=ON');conn.execute('BEGIN')
            marks=','.join('?' for _ in tag_ids)
            docs={r['id']:r['updated_at'] for r in conn.execute(f'''SELECT id,updated_at FROM documents
                WHERE id IN (SELECT document_id FROM document_tags WHERE tag_id IN ({marks}))''',tag_ids)}
            cache={};valid=[]
            for candidate in candidates:
                if docs.get(candidate['document_id'])!=candidate['updated_at']:continue
                accepted=True
                for segment in ranges(candidate):
                    key=(candidate['document_id'],segment['block_id'])
                    if key not in cache:
                        row=conn.execute('SELECT text_content FROM content_blocks WHERE document_id=? AND id=?',key).fetchone()
                        cache[key]=sha256(row['text_content'].encode()).hexdigest() if row else None
                    if cache[key]!=segment['content_hash']:accepted=False;break
                if accepted:valid.append(candidate)
            return valid

    def search(self,question,group_ids,tag_ids,expansion=None):
        expansion=expansion or {'state':'local','queries':[],'terms':[]}
        result=self.keyword.search(question,group_ids,tag_ids,limit=120,allow_empty_terms=True,
            extra_terms=[*expansion['terms'],*expansion.get('local_terms',[])])
        documents=resolve_documents(self.db_service,group_ids,tag_ids)
        rewritten=question
        for group in result.get('query_groups',[]):
            if group['aliases']:rewritten=rewritten.replace(group['term'],group['aliases'][0])
        groups=result.get('query_groups',[])
        focused=max(groups,key=lambda g:g['weight'],default=None)
        subject=(focused['aliases'] or [focused['term']])[0] if focused else question
        english_queries=expansion['queries']
        focused_queries=list(dict.fromkeys([question,*english_queries,*expansion.get('local_queries',[]),rewritten]))
        queries=list(dict.fromkeys([*focused_queries,subject,
            *(subject+' '+term for term in result.get('related_terms',[]))]))[:10]
        try:
            by_chunk={};truncated=False
            if callable(getattr(self.index,'search_many',None)):
                batches,status,truncated=self.index.search_many(queries,documents,limit=60)
            else:
                batches=[]
                for query in queries:
                    hits,status,cut=self.index.search(query,documents,limit=60)
                    batches.append(hits);truncated=truncated or cut
            for query_index,hits in enumerate(batches):
                for fallback_rank,item in enumerate(hits,1):
                    # Full question has priority over broad subject/facet variants.
                    penalty=0 if query_index<=len(english_queries) else (10 if queries[query_index] in focused_queries else 45)
                    rank=item.get('language_rank',fallback_rank)+penalty
                    previous=by_chunk.get(item['chunk_id'])
                    if previous and previous['semantic_rank']<=rank:continue
                    by_chunk[item['chunk_id']]=dict(item,semantic_rank=rank)
            semantic=list(by_chunk.values())
            semantic.sort(key=lambda c:(c['semantic_rank'],-c['semantic_score']))
        except Exception as error:
            semantic=[];status={**self.index.status(documents),'state':'error','error':str(error)[:400]};truncated=False
        from app.research_evidence import ContextExpander, relevance, text_language
        groups=result.get('query_groups',[])
        with closing(self.db_service.get_connection()) as conn:
            conn.execute('PRAGMA query_only=ON');conn.execute('BEGIN')
            expander=ContextExpander(conn)
            semantic=[expander.expand(item) for item in semantic]
        keywords=result['citations'];merged=[];covered=set()
        for rank,item in enumerate(semantic,1):
            item=dict(item);item['search_methods']=['semantic'];score=1/(60+item.get('semantic_rank',rank))
            matching=[(i,k) for i,k in enumerate(keywords,1) if overlaps(item,k,.5)
                      and any(term.lower() in item['quote'].lower() for term in k['matched_terms'])]
            if matching:
                score+=1/(60+matching[0][0]);item['search_methods'].append('keyword')
                covered.update(i for i,k in matching)
            item['_fusion']=score;merged.append(item)
        for rank,item in enumerate(keywords,1):
            if rank not in covered:merged.append({**item,'search_methods':['keyword'],'_fusion':1/(60+rank)})
        # Score languages against their own query vocabulary. Additional English
        # terms must not dilute the score of a strong Japanese original passage.
        from app.research_evidence import INTENT_WORDS
        from app.research_search import query_terms
        originals=set(query_terms(question))
        subject_count=max(1,sum(g['term'] in originals and g['term'] not in INTENT_WORDS for g in groups))
        def normalized(text,vocabulary):
            score,_,_=relevance(text,vocabulary)
            return score/(sum(g['weight'] for g in vocabulary) or 1)
        def topicality(item):
            lane=item['retrieval_language']
            applicable=[g for g in groups if g['term'] not in INTENT_WORDS
                        and (text_language(g['term']*20)==lane)]
            if not applicable:return 0.
            primary=[g for g in applicable if g['term'] in originals]
            variants=[g for g in applicable if g['term'] not in originals]
            # Synonyms are alternatives, not additional required subjects. A
            # direct match to the full original question must keep its strength
            # when many translated/alternate terms were added by the planner.
            if primary:return max(normalized(item['quote'],primary),.85*normalized(item['quote'],variants))
            # For translated queries, use the query's subjects, not the entire
            # synonym list as a checklist of topics every passage must contain.
            translated=[]
            for query in english_queries:
                vocabulary=[g for g in applicable if g['term'].lower() in query.lower()]
                if not vocabulary:continue
                score,_,_=relevance(item['quote'],vocabulary)
                weights=sorted((g['weight'] for g in vocabulary),reverse=True)[:subject_count]
                translated.append(min(1.,score/(sum(weights) or 1)))
            return max(translated,default=normalized(item['quote'],variants))
        doc_support={}
        for item in merged:
            if 'retrieval_language' not in item:item['retrieval_language']=text_language(item['quote'])
            doc_support[item['document_id']]=max(doc_support.get(item['document_id'],0),topicality(item))
        for item in merged:
            score,coverage,found=relevance(item['quote'],groups)
            item['matched_terms']=found
            topic=topicality(item)
            # A semantic rank by itself is weak evidence for relevance. Keep it
            # available for synonyms, but prefer passages corroborated by query
            # vocabulary and other relevant passages in the same document.
            item['_relevance']=item['_fusion']*(.3+.7*topic)+.020*topic+.006*doc_support[item['document_id']]
        remaining=self.fresh(merged,tag_ids);candidates=[];doc_counts={};section_counts={}
        while remaining and len(candidates)<32:
            # Reserve English opportunities only when their relevance is close
            # to the best remaining passage. Language alone never fills a slot.
            def utility(c):
                section=(c['document_id'],c.get('section_title',''))
                return c['_relevance']/(1+.04*doc_counts.get(c['document_id'],0)+.30*section_counts.get(section,0))
            best=max(remaining,key=utility)
            english=[c for c in remaining if c['retrieval_language']=='en' and utility(c)>=.85*utility(best)]
            item=max(english,key=utility) if len(candidates)%4==3 and english else best
            remaining.remove(item)
            if any(overlaps(item,other,.5) for other in candidates):continue
            item.pop('_fusion',None);item.pop('_relevance',None)
            item['id']=str(len(candidates)+1);candidates.append(item)
            doc_counts[item['document_id']]=doc_counts.get(item['document_id'],0)+1
            section=(item['document_id'],item.get('section_title',''))
            section_counts[section]=section_counts.get(section,0)+1
        result.update(citations=candidates,limit=32,document_count=len(documents),
                      retrieval='local-hybrid-depth-v1' if semantic else 'local-keyword-depth-v1',
                      semantic={k:v for k,v in status.items() if k!='ready_ids'},
                      embedding_model=self.index.provider.model_key,query_truncated=truncated)
        result['retrieval_queries']=queries
        result['cross_language_search']=bool(semantic)
        result['english_candidates']=sum(c.get('retrieval_language')=='en' for c in candidates)
        result['english_query_expansion']={k:v for k,v in expansion.items() if k!='provider'}
        return result
