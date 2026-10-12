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
        result=self.keyword.search(question,group_ids,tag_ids,limit=80,allow_empty_terms=True,extra_terms=expansion['terms'])
        documents=resolve_documents(self.db_service,group_ids,tag_ids)
        rewritten=question
        for group in result.get('query_groups',[]):
            if group['aliases']:rewritten=rewritten.replace(group['term'],group['aliases'][0])
        groups=result.get('query_groups',[])
        focused=max(groups,key=lambda g:g['weight'],default=None)
        subject=(focused['aliases'] or [focused['term']])[0] if focused else question
        english_queries=expansion['queries']
        queries=list(dict.fromkeys([question,*english_queries,rewritten,subject,
            *(subject+' '+term for term in result.get('related_terms',[]))]))[:6]
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
                    rank=item.get('language_rank',fallback_rank)+(0 if query_index<=len(english_queries) else 30)
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
        weight=sum(g['weight'] for g in groups) or 1
        for item in merged:
            if 'retrieval_language' not in item:item['retrieval_language']=text_language(item['quote'])
            score,coverage,found=relevance(item['quote'],groups)
            item['matched_terms']=found
            item['_relevance']=item['_fusion']+(.008*score/weight if 'semantic' in item['search_methods'] else .012*score/weight)
        remaining=self.fresh(merged,tag_ids);candidates=[];doc_counts={}
        while remaining and len(candidates)<32:
            # Keep English candidates available within the bounded answer budget.
            # Generation still chooses evidence by relevance, never by language quota.
            lane='en' if len(candidates)%4==3 else 'ja'
            pool=[c for c in remaining if c['retrieval_language']==lane]
            item=max(pool or remaining,key=lambda c:c['_relevance']/(1+.04*doc_counts.get(c['document_id'],0)))
            remaining.remove(item)
            if any(overlaps(item,other,.5) for other in candidates):continue
            item.pop('_fusion',None);item.pop('_relevance',None)
            item['id']=str(len(candidates)+1);candidates.append(item)
            doc_counts[item['document_id']]=doc_counts.get(item['document_id'],0)+1
        result.update(citations=candidates,limit=32,document_count=len(documents),
                      retrieval='local-hybrid-multilingual-v1' if semantic else 'local-keyword-context-v2',
                      semantic={k:v for k,v in status.items() if k!='ready_ids'},
                      embedding_model=self.index.provider.model_key,query_truncated=truncated)
        result['retrieval_queries']=queries
        result['cross_language_search']=bool(semantic)
        result['english_candidates']=sum(c.get('retrieval_language')=='en' for c in candidates)
        result['english_query_expansion']={k:v for k,v in expansion.items() if k!='provider'}
        return result
