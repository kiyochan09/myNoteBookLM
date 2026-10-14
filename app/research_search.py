"""Local evidence retrieval and immutable search history; generation is separate."""
from contextlib import closing
from hashlib import sha256
import json
import re
import sqlite3
import time
from typing import Protocol

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    id: str = Field(min_length=1, max_length=100)
    revision: int = Field(ge=1)
    question: str = Field(min_length=1, max_length=12000)
    context_answer_id: str | None = Field(default=None, max_length=100)
    answer_preparation: bool = False


def query_terms(question):
    # Quoted phrases are preserved; Japanese questions also use script boundaries.
    phrases = re.findall(r'[「『"]([^」』"\n]+)[」』"]', question)
    cleaned = re.sub(r'[「『"][^」』"\n]+[」』"]', ' ', question)
    stop = {'選択', '資料', '文書', '出典', '引用', '箇所', '記述', '関係', '質問',
            '整理', '比較', '説明', '中心的', '論点', '共通', '異なる', '見解', '頁', 'ページ', '教', '集', '付'}
    pieces = re.findall(r'[一-龯々〆ヵヶ]+|[ァ-ヴー]{2,}|[A-Za-z0-9][A-Za-z0-9_.-]*', cleaned)
    terms = []
    for term in phrases + pieces:
        term = term.strip()[:120]
        if term and term not in stop and term.lower() not in {t.lower() for t in terms}:
            terms.append(term)
    return terms[:16]


class EvidenceRetriever(Protocol):
    def search(self, question: str, group_ids: list, tag_ids: list) -> dict: ...


class LocalEvidenceRetriever:
    def __init__(self, db_service):
        self.db_service = db_service

    def search(self, question, group_ids, tag_ids, limit=20, allow_empty_terms=False, extra_terms=None):
        terms = list(dict.fromkeys(query_terms(question)+list(extra_terms or [])))
        if not terms and not allow_empty_terms:
            raise HTTPException(400, '探したい言葉を具体的に入力してください。語句を「 」で囲むと、そのまま検索できます。')
        if not group_ids or not tag_ids:
            raise HTTPException(400, '検索するグループとタグを選んでください。')
        with closing(self.db_service.get_connection()) as conn:
            conn.execute('PRAGMA query_only=ON')
            deadline = time.monotonic() + 20
            conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
            conn.execute('BEGIN')
            groups = {r['id'] for r in conn.execute('SELECT id FROM tag_groups')}
            tags = {r['id']: r['group_id'] for r in conn.execute('SELECT id,group_id FROM tags')}
            if not set(group_ids) <= groups or any(tags.get(t) not in group_ids for t in tag_ids):
                raise HTTPException(400, '資料のタグが変更されています。資料一覧を更新してください。')
            placeholders = ','.join('?' for _ in tag_ids)
            scope = f'SELECT DISTINCT document_id FROM document_tags WHERE tag_id IN ({placeholders})'
            count = conn.execute(f'SELECT count(*) FROM documents WHERE id IN ({scope})', tag_ids).fetchone()[0]
            rows = {}
            columns = '''cb.id AS block_id,cb.document_id,cb.text_content,cb.page_number,
                cb.reading_order,d.title,d.updated_at,COALESCE(s.title,'') AS section_title'''
            joins = '''JOIN documents d ON d.id=cb.document_id LEFT JOIN sections s ON s.id=cb.section_id'''
            from app.research_evidence import lexical_groups, relevance, related_terms, ContextExpander, evidence_windows
            cache = {}
            # Read the selected corpus once. Global FTS statistics/joins can be
            # very expensive for dozens of expansion terms in a large library.
            # BM25 term-frequency ranking below is scoped before any limit.
            corpus=[dict(r) for r in conn.execute(f'''SELECT {columns} FROM content_blocks cb {joins}
                WHERE cb.document_id IN ({scope}) AND length(trim(cb.text_content))>0''',tag_ids)]
            average_length=sum(len(r['text_content']) for r in corpus)/max(1,len(corpus))
            for row in corpus:row['_lower']=row['text_content'].lower()
            def lookup(term):
                if term in cache:return cache[term]
                if time.monotonic()>deadline:raise HTTPException(408,'検索に時間がかかっています。タグを絞って再検索してください。')
                matches=[row for row in corpus if term.lower() in row['_lower']]
                frequency=len({row['document_id'] for row in matches})
                def score(row):
                    tf=row['_lower'].count(term.lower())
                    return -tf/(tf+1.2*(.25+.75*len(row['text_content'])/max(1,average_length)))
                matches.sort(key=lambda r:(score(r),r['document_id'],r['reading_order'],r['block_id']))
                # Preserve a few strong passages from every scoped document before
                # the global cutoff, so a repetitive book cannot hide another book.
                chosen=matches[:96];seen={r['block_id'] for r in chosen};per_doc={}
                for row in matches:
                    doc=row['document_id'];per_doc[doc]=per_doc.get(doc,0)+1
                    if per_doc[doc]<=4 and row['block_id'] not in seen:
                        chosen.append(row);seen.add(row['block_id'])
                found=[{k:v for k,v in row.items() if k!='_lower'}|{'score':score(row)} for row in chosen]
                cache[term] = (found, frequency)
                return cache[term]
            try:
                query_groups = lexical_groups(question, terms, count, lookup)
                for group in query_groups:
                    for term in [group['term'], *group['aliases']]:
                        for row in lookup(term)[0]:rows[row['block_id']] = row
                facets=related_terms(query_groups,list(rows.values()),count,lookup)
            except sqlite3.OperationalError as error:
                if 'interrupt' in str(error):
                    raise HTTPException(408, '検索に時間がかかっています。タグを絞るか、より長い語句で再検索してください。') from error
                raise
            ranked = []
            for row in rows.values():
                text = row['text_content'] or ''
                for start,end in evidence_windows(text,query_groups):
                    score,coverage,found=relevance(text[start:end],query_groups)
                    ranked.append(dict(row,matched_terms=found,start=start,end=end,quote=text[start:end],
                        content_hash=sha256(text.encode()).hexdigest(),truncated=start>0 or end<len(text),
                        lexical_score=score,lexical_coverage=coverage))
            ranked.sort(key=lambda r: (-r['lexical_score'], -r['lexical_coverage'],len(r['quote']) < 40,
                                       r['score'],r['document_id'],r['reading_order'],r['block_id']))
            expander = ContextExpander(conn)
            citations = [];doc_counts={}
            from app.research_hybrid import overlaps
            while ranked:
                row=max(ranked,key=lambda r:r['lexical_score']/(1+.12*doc_counts.get(r['document_id'],0)))
                ranked.remove(row)
                row = dict(row);row.pop('text_content');row.pop('score')
                row = expander.expand(row)
                score,coverage,found=relevance(row['quote'],query_groups)
                row.update(lexical_score=score,lexical_coverage=coverage,matched_terms=found)
                if any(overlaps(row,other,.5) for other in citations):continue
                row['id'] = str(len(citations)+1);citations.append(row)
                doc_counts[row['document_id']]=doc_counts.get(row['document_id'],0)+1
                if len(citations) == limit:break
            tag_names = [r['name'] for r in conn.execute(f'SELECT name FROM tags WHERE id IN ({placeholders}) ORDER BY name',tag_ids)]
            return dict(question=question, terms=terms, document_count=count, groupIds=group_ids, tag_names=tag_names,
                        tagIds=tag_ids, citations=citations, limit=limit,
                        retrieval='local-keyword-depth-v1', candidate_limit=sum(len(v[0]) for v in cache.values()), query_groups=query_groups, related_terms=facets)


def create_search_router(path, db_service, semantic_index=None):
    from app.research_threads import ResearchThreads
    store = ResearchThreads(path)
    router = APIRouter(prefix='/api/research/threads')
    retriever: EvidenceRetriever = LocalEvidenceRetriever(db_service)
    if semantic_index is not None:
        from app.research_hybrid import HybridEvidenceRetriever
        retriever = HybridEvidenceRetriever(retriever,semantic_index,db_service)
    with store.connect() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS research_searches (
            id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, created_at TEXT NOT NULL,
            question TEXT NOT NULL, result_json TEXT NOT NULL)''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_research_search_thread ON research_searches(thread_id,created_at DESC)')

    def thread(conn, thread_id, revision=None):
        row = conn.execute('SELECT * FROM research_threads WHERE id=? AND deleted_at IS NULL', (thread_id,)).fetchone()
        if not row:
            raise HTTPException(404, 'スレッドが見つかりません。')
        if revision is not None and row['revision'] != revision:
            raise HTTPException(409, 'スレッドが変更されています。再読み込みしてから検索してください。')
        return store.decode(row)

    def get_run(thread_id, run_id):
        with store.connect() as conn:
            thread(conn, thread_id)
            row = conn.execute('SELECT * FROM research_searches WHERE thread_id=? AND id=?', (thread_id,run_id)).fetchone()
            if not row:
                raise HTTPException(404, '検索結果が見つかりません。')
            return json.loads(row['result_json'])

    if semantic_index is not None:
        from app.research_index import resolve_documents

        class IndexRequest(BaseModel):
            revision: int = Field(ge=1)
            retry: bool = False

        def scope(thread_id,revision=None):
            with store.connect() as conn:selected=thread(conn,thread_id,revision)
            return resolve_documents(db_service,selected['groupIds'],selected['tagIds'])

        def public_status(result):return {k:v for k,v in result.items() if k!='ready_ids'}

        @router.get('/{thread_id}/index')
        def index_status(thread_id:str):
            return {'index':public_status(semantic_index.status(scope(thread_id)))}

        @router.post('/{thread_id}/index')
        def prepare_index(thread_id:str,data:IndexRequest):
            return {'index':public_status(semantic_index.prepare(scope(thread_id,data.revision),retry=data.retry))}

        @router.delete('/{thread_id}/index')
        def pause_index(thread_id:str):
            return {'index':public_status(semantic_index.pause(scope(thread_id)))}

    @router.get('/{thread_id}/searches')
    def list_searches(thread_id: str):
        with store.connect() as conn:
            thread(conn, thread_id)
            return {'searches': [dict(row) for row in conn.execute('''SELECT id,question,created_at
                FROM research_searches WHERE thread_id=? ORDER BY created_at DESC,id DESC LIMIT 50''', (thread_id,))]}

    @router.get('/{thread_id}/searches/{run_id}')
    def search_detail(thread_id: str, run_id: str):
        return {'search': get_run(thread_id, run_id)}

    @router.post('/{thread_id}/searches')
    def search(thread_id: str, data: SearchRequest):
        question = data.question.strip()
        if not question:
            raise HTTPException(400, '質問や検索する語句を入力してください。')
        with store.connect() as conn:
            selected = thread(conn, thread_id, data.revision)
            existing = conn.execute('SELECT thread_id,question,result_json FROM research_searches WHERE id=?', (data.id,)).fetchone()
            if existing:
                if existing['thread_id'] != thread_id or existing['question'] != question or json.loads(existing['result_json']).get('context_answer_id') != data.context_answer_id:
                    raise HTTPException(409, '検索IDが重複しています。再検索してください。')
                return {'search': json.loads(existing['result_json'])}
        from app.research_answers import get_context
        context = get_context(store, thread_id, data.context_answer_id, selected['tagIds'])
        retrieval_question = (context['question'] + '\n' + question) if context else question
        expansion=None
        if data.answer_preparation and semantic_index is not None:
            from app.research_ai_settings import AISettings
            from app.research_query_expansion import GeminiQueryExpansion
            from pathlib import Path
            try:
                config,key=AISettings(Path(path).parent/'research_ai.json').credentials()
                if config['enabled'] and key:
                    expansion=GeminiQueryExpansion(key,config['model']).expand(retrieval_question)
                else:expansion={'state':'disabled','queries':[],'terms':[]}
            except Exception:
                # Never leak a transport exception containing a credential or question.
                expansion={'state':'unavailable','queries':[],'terms':[]}
        result = (retriever.search(retrieval_question, selected['groupIds'], selected['tagIds'],expansion=expansion)
                  if semantic_index is not None else retriever.search(retrieval_question,selected['groupIds'],selected['tagIds']))
        result['answer_preparation']=data.answer_preparation
        result.update(question=question, context_answer_id=data.context_answer_id, retrieval_question=retrieval_question)
        result.update(id=data.id, thread_id=thread_id, created_at=store.now())
        with store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            thread(conn, thread_id, data.revision)  # Do not attach results to a changed/deleted scope.
            conn.execute('INSERT OR IGNORE INTO research_searches VALUES (?,?,?,?,?)',
                         (data.id,thread_id,result['created_at'],question,json.dumps(result,ensure_ascii=False)))
        return {'search': get_run(thread_id,data.id)}

    @router.get('/{thread_id}/searches/{run_id}/sources/{citation_id}')
    def source(thread_id: str, run_id: str, citation_id: str):
        run = get_run(thread_id,run_id)
        saved = next((c for c in run['citations'] if c['id']==citation_id),None)
        if not saved:
            raise HTTPException(404, '出典が見つかりません。')
        with closing(db_service.get_connection()) as conn:
            conn.execute('PRAGMA query_only=ON')
            doc = conn.execute('SELECT title,updated_at FROM documents WHERE id=?', (saved['document_id'],)).fetchone()
            if not doc:
                return {'source':saved, 'state':'deleted', 'message':'元文書が削除されています。検索時の引用を表示しています。'}
            unchanged = True
            for segment in saved.get('segments') or [saved]:
                block = conn.execute('SELECT text_content FROM content_blocks WHERE id=? AND document_id=?',
                                     (segment['block_id'],saved['document_id'])).fetchone()
                if not block or sha256(block['text_content'].encode()).hexdigest()!=segment['content_hash']:
                    unchanged = False;break
            state = 'current' if unchanged and doc['updated_at']==saved['updated_at'] else 'changed'
            return {'source':saved, 'state':state, 'current_title':doc['title'],
                    'message':'検索時の引用です。元文書は検索後に更新されています。' if state=='changed' else '保存本文からの引用です。'}
    return router
