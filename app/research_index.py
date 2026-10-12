"""Scoped, resumable local passage index. Source documents are read-only."""
from contextlib import closing,contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import queue
import sqlite3
import threading
import time

from fastapi import HTTPException


def resolve_documents(db_service,group_ids,tag_ids):
    if not group_ids or not tag_ids:return []
    with closing(db_service.get_connection()) as conn:
        conn.execute('PRAGMA query_only=ON');conn.execute('BEGIN')
        groups={r['id'] for r in conn.execute('SELECT id FROM tag_groups')}
        tags={r['id']:r['group_id'] for r in conn.execute('SELECT id,group_id FROM tags')}
        if not set(group_ids)<=groups or any(tags.get(t) not in group_ids for t in tag_ids):
            raise HTTPException(400,'資料のタグが変更されています。資料一覧を更新してください。')
        marks=','.join('?' for _ in tag_ids)
        return [dict(r) for r in conn.execute(f'''SELECT id,title,updated_at FROM documents
            WHERE id IN (SELECT document_id FROM document_tags WHERE tag_id IN ({marks})) ORDER BY id''',tag_ids)]


def read_document(db_service,doc_id):
    with closing(db_service.get_connection()) as conn:
        conn.execute('PRAGMA query_only=ON');conn.execute('BEGIN')
        doc=conn.execute('SELECT id,title,updated_at FROM documents WHERE id=?',(doc_id,)).fetchone()
        if not doc:return None
        rows=[dict(r) for r in conn.execute('''SELECT cb.id AS block_id,cb.text_content,cb.reading_order,
            cb.page_number,cb.section_id,COALESCE(s.title,'') AS section_title
            FROM content_blocks cb LEFT JOIN sections s ON s.id=cb.section_id
            WHERE cb.document_id=? ORDER BY cb.reading_order,cb.id''',(doc_id,))]
    doc=dict(doc);digest=sha256(json.dumps(doc,ensure_ascii=False,sort_keys=True).encode())
    for row in rows:
        row['content_hash']=sha256(row['text_content'].encode()).hexdigest()
        digest.update(json.dumps(row,ensure_ascii=False,sort_keys=True).encode())
    return {'document':doc,'blocks':rows,'fingerprint':digest.hexdigest()}


def passage_chunks(snapshot,provider):
    doc=snapshot['document'];groups=[];current=[];key=None
    for block in snapshot['blocks']:
        if not block['text_content'].strip():continue
        next_key=(block['page_number'],block['section_id'])
        if current and key!=next_key:groups.append(current);current=[]
        current.append(block);key=next_key
    if current:groups.append(current)
    for group in groups:
        seen=set()
        text='';spans=[]
        for block in group:
            if text:text+='\n'
            start=len(text);text+=block['text_content'];spans.append((start,len(text),block))
        prefix='passage: '+doc['title'][:80]+'\n'+group[0]['section_title'][:80]+'\n'
        base=0
        while base<len(text):
            finish=min(len(text),base+1000)
            for left,right in provider.windows(text[base:finish],prefix):
                left+=base;right+=base
                if (left,right) in seen:continue
                seen.add((left,right))
                segments=[]
                for start,end,block in spans:
                    if end>left and start<right:
                        segments.append({'block_id':block['block_id'],'content_hash':block['content_hash'],
                                         'start':max(left,start)-start,'end':min(right,end)-start})
                if not segments:continue
                first=segments[0]
                yield {'document_id':doc['id'],'title':doc['title'],'updated_at':doc['updated_at'],
                       'page_number':group[0]['page_number'],'section_title':group[0]['section_title'],
                       'block_id':first['block_id'],'content_hash':first['content_hash'],
                       'start':first['start'],'end':first['end'],'segments':segments,
                       'reading_order':next(b['reading_order'] for b in group if b['block_id']==first['block_id']),
                       'quote':text[left:right],'has_prefix':left>0,'has_suffix':right<len(text),
                       'truncated':left>0 or right<len(text),'embedding_text':prefix+text[left:right]}
            if finish==len(text):break
            base=finish-160


class SemanticIndex:
    def __init__(self,path,db_service,provider):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db_service=db_service;self.provider=provider
        self.lock=threading.Lock();self.jobs=queue.PriorityQueue();self.pending=set();self.canceled=set()
        self.priorities={};self.job_counter=0;self.active_job=None
        self.stop=threading.Event();self.worker=None
        with self.connect() as conn:
            conn.executescript('''CREATE TABLE IF NOT EXISTS semantic_documents (
                doc_id TEXT PRIMARY KEY,model_key TEXT NOT NULL,fingerprint TEXT NOT NULL,
                source_updated TEXT NOT NULL,source_title TEXT NOT NULL,state TEXT NOT NULL,
                completed INTEGER NOT NULL,total INTEGER NOT NULL,error TEXT NOT NULL,updated_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS semantic_chunks (
                id TEXT PRIMARY KEY,doc_id TEXT NOT NULL,model_key TEXT NOT NULL,fingerprint TEXT NOT NULL,
                payload_json TEXT NOT NULL,vector BLOB NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_semantic_chunks_scope ON semantic_chunks(doc_id,model_key,fingerprint);''')

    @contextmanager
    def connect(self):
        conn=sqlite3.connect(str(self.path),timeout=15);conn.row_factory=sqlite3.Row
        try:
            with conn:yield conn
        finally:conn.close()

    def records(self,ids):
        if not ids:return {}
        with self.connect() as conn:
            marks=','.join('?' for _ in ids)
            return {r['doc_id']:dict(r) for r in conn.execute(f'SELECT * FROM semantic_documents WHERE doc_id IN ({marks})',ids)}

    def status(self,documents,fingerprints=None):
        records=self.records([d['id'] for d in documents]);ready=[];states=[];completed=0;total=0;errors=[]
        for doc in documents:
            row=records.get(doc['id'])
            fresh=row and row['model_key']==self.provider.model_key and row['source_updated']==doc['updated_at'] and row['source_title']==doc['title']
            if fresh and fingerprints is not None:fresh=row['fingerprint']==fingerprints.get(doc['id'])
            state=row['state'] if fresh else 'building' if doc['id'] in self.pending else 'missing';states.append(state)
            if fresh:
                completed+=row['completed'];total+=row['total']
                if state=='ready':ready.append(doc['id'])
                if row['error']:errors.append(row['error'])
        state='ready' if len(ready)==len(documents) else 'building' if any(s in {'building','queued'} for s in states) else 'error' if 'error' in states else 'paused' if 'paused' in states else 'missing'
        return {'state':state,'ready_documents':len(ready),'total_documents':len(documents),
                'completed_chunks':completed,'total_chunks':total,'error':errors[0] if errors else '',
                'model':'multilingual-e5-small','provider':'local-onnx-cpu','ready_ids':ready}

    def prepare(self,documents,retry=False,fingerprints=None):
        self.prune_deleted()
        records=self.records([d['id'] for d in documents])
        with self.lock:
            self.job_counter+=1;priority=-self.job_counter
            for doc in documents:
                row=records.get(doc['id'])
                fresh=row and row['model_key']==self.provider.model_key and row['source_updated']==doc['updated_at'] and row['source_title']==doc['title']
                if fresh and fingerprints is not None:fresh=row['fingerprint']==fingerprints.get(doc['id'])
                if fresh and row['state']=='ready':continue
                if fresh and row['state'] in {'paused','error'} and not retry:continue
                if doc['id'] in self.pending:
                    if retry:
                        self.canceled.discard(doc['id'])
                        with self.connect() as conn:conn.execute("UPDATE semantic_documents SET state='building' WHERE doc_id=? AND state='paused'",(doc['id'],))
                    if doc['id']!=self.active_job:
                        self.priorities[doc['id']]=priority;self.jobs.put((priority,doc['id']))
                    continue
                self.canceled.discard(doc['id']);self.pending.add(doc['id'])
                self.priorities[doc['id']]=priority
                with self.connect() as conn:
                    conn.execute('''INSERT INTO semantic_documents VALUES (?,?,?,?,?,'queued',0,0,'',?)
                        ON CONFLICT(doc_id) DO UPDATE SET state='queued',error='',updated_at=excluded.updated_at''',
                        (doc['id'],self.provider.model_key,'',doc['updated_at'],doc['title'],time.time()))
                self.jobs.put((priority,doc['id']))
            if self.pending and not (self.worker and self.worker.is_alive()):
                self.worker=threading.Thread(target=self.work,name='research-local-index',daemon=True);self.worker.start()
        return self.status(documents,fingerprints)

    def prune_deleted(self):
        with closing(self.db_service.get_connection()) as source:
            source.execute('PRAGMA query_only=ON')
            existing={r['id'] for r in source.execute('SELECT id FROM documents')}
        with self.connect() as conn:
            deleted=[r['doc_id'] for r in conn.execute('SELECT doc_id FROM semantic_documents') if r['doc_id'] not in existing]
            for doc_id in deleted:
                conn.execute('DELETE FROM semantic_chunks WHERE doc_id=?',(doc_id,))
                conn.execute('DELETE FROM semantic_documents WHERE doc_id=?',(doc_id,))

    def pause(self,documents):
        with self.lock:self.canceled.update(d['id'] for d in documents)
        with self.connect() as conn:
            for doc in documents:
                conn.execute("UPDATE semantic_documents SET state='paused',updated_at=? WHERE doc_id=? AND state!='ready'",(time.time(),doc['id']))
        return self.status(documents)

    @contextmanager
    def writer_lock(self):
        # A reload or second server must not run two index writers concurrently.
        path=self.path.with_suffix('.lock');handle=path.open('a+b');handle.seek(0)
        if handle.read(1)==b'':handle.write(b'0');handle.flush()
        acquired=False
        try:
            while not self.stop.is_set():
                try:
                    handle.seek(0)
                    if os.name=='nt':
                        import msvcrt
                        msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                    else:
                        import fcntl
                        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    acquired=True;break
                except OSError:self.stop.wait(.25)
            yield acquired
        finally:
            if acquired:
                handle.seek(0)
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
                else:
                    import fcntl
                    fcntl.flock(handle,fcntl.LOCK_UN)
            handle.close()

    def work(self):
        while not self.stop.is_set():
            try:priority,doc_id=self.jobs.get(timeout=.5)
            except queue.Empty:continue
            with self.lock:
                if self.priorities.get(doc_id)!=priority or doc_id not in self.pending:
                    self.jobs.task_done();continue
                self.active_job=doc_id
            try:
                with self.writer_lock() as acquired:
                    if acquired:self.index_document(doc_id)
            except Exception as error:
                if not self.stop.is_set():
                    with self.connect() as conn:conn.execute("UPDATE semantic_documents SET state='error',error=?,updated_at=? WHERE doc_id=?",(str(error)[:400],time.time(),doc_id))
            finally:
                with self.lock:
                    if doc_id in self.canceled and not self.stop.is_set():
                        with self.connect() as conn:conn.execute("UPDATE semantic_documents SET state='paused' WHERE doc_id=? AND state!='ready'",(doc_id,))
                    self.pending.discard(doc_id)
                    self.priorities.pop(doc_id,None);self.active_job=None
                self.jobs.task_done()

    def interrupted(self,doc_id):return self.stop.is_set() or doc_id in self.canceled

    def index_document(self,doc_id):
        if self.interrupted(doc_id):return
        snapshot=read_document(self.db_service,doc_id)
        if snapshot is None:
            with self.connect() as conn:
                conn.execute('DELETE FROM semantic_chunks WHERE doc_id=?',(doc_id,));conn.execute('DELETE FROM semantic_documents WHERE doc_id=?',(doc_id,))
            return
        doc=snapshot['document'];fingerprint=snapshot['fingerprint'];key=self.provider.model_key
        with self.connect() as conn:
            row=conn.execute('SELECT * FROM semantic_documents WHERE doc_id=?',(doc_id,)).fetchone()
            if row and row['state']=='ready' and row['model_key']==key and row['fingerprint']==fingerprint:return
            conn.execute('''UPDATE semantic_documents SET model_key=?,fingerprint=?,source_updated=?,source_title=?,state='building',completed=0,total=0,error='',updated_at=? WHERE doc_id=?''',
                         (key,fingerprint,doc['updated_at'],doc['title'],time.time(),doc_id))
        chunks=[]
        for chunk in passage_chunks(snapshot,self.provider):
            if self.interrupted(doc_id):return
            identity=json.dumps([key,doc_id,fingerprint,chunk['segments']],sort_keys=True)
            chunk['chunk_id']=sha256(identity.encode()).hexdigest();chunks.append(chunk)
        with self.connect() as conn:
            cached={r['id'] for r in conn.execute('SELECT id FROM semantic_chunks WHERE doc_id=? AND model_key=? AND fingerprint=?',(doc_id,key,fingerprint))}
            conn.execute('UPDATE semantic_documents SET total=?,completed=?,updated_at=? WHERE doc_id=?',(len(chunks),len(cached),time.time(),doc_id))
        remaining=[c for c in chunks if c['chunk_id'] not in cached];completed=len(cached)
        for start in range(0,len(remaining),4):
            if self.interrupted(doc_id):return
            batch=remaining[start:start+4]
            embedded=self.provider.embed([c['embedding_text'] for c in batch],'passage')
            with self.connect() as conn:
                for chunk,vector in zip(batch,embedded['vectors'],strict=True):
                    payload={k:v for k,v in chunk.items() if k!='embedding_text'}
                    conn.execute('INSERT OR REPLACE INTO semantic_chunks VALUES (?,?,?,?,?,?)',
                                 (chunk['chunk_id'],doc_id,key,fingerprint,json.dumps(payload,ensure_ascii=False),vector))
                completed+=len(batch)
                conn.execute('UPDATE semantic_documents SET completed=?,updated_at=? WHERE doc_id=?',(completed,time.time(),doc_id))
        if self.interrupted(doc_id):return
        current=read_document(self.db_service,doc_id)
        if current is None:
            with self.connect() as conn:
                conn.execute('DELETE FROM semantic_chunks WHERE doc_id=?',(doc_id,));conn.execute('DELETE FROM semantic_documents WHERE doc_id=?',(doc_id,))
            return
        if current['fingerprint']!=fingerprint:
            with self.connect() as conn:conn.execute("UPDATE semantic_documents SET state='missing',error='準備中に本文が更新されました。再度準備してください。' WHERE doc_id=?",(doc_id,))
            return
        with self.connect() as conn:
            conn.execute("UPDATE semantic_documents SET state='ready',completed=total,error='',updated_at=? WHERE doc_id=?",(time.time(),doc_id))
            conn.execute('DELETE FROM semantic_chunks WHERE doc_id=? AND (model_key!=? OR fingerprint!=?)',(doc_id,key,fingerprint))

    def search(self,question,documents,limit=40):
        batches,status,truncated=self.search_many([question],documents,limit=limit)
        return batches[0],status,truncated

    def search_many(self,questions,documents,limit=40):
        if not questions:return [],self.status(documents),False
        # One verified source snapshot and one vector scan for all query variants.
        # Still detect same-timestamp replacements; do not cache across searches.
        fingerprints={};document_languages={}
        for document in documents:
            snapshot=read_document(self.db_service,document['id'])
            if snapshot:
                fingerprints[document['id']]=snapshot['fingerprint']
                from app.research_evidence import text_language
                blocks=snapshot['blocks'];step=max(1,len(blocks)//20)
                sample='\n'.join(block['text_content'][::max(1,len(block['text_content'])//1000)]
                                 for block in blocks[::step])
                document_languages[document['id']]=text_language(sample)
        status=self.prepare(documents,fingerprints=fingerprints)
        ready=set(status['ready_ids'])
        candidates=[{'ja':[], 'en':[]} for _ in questions]
        if not ready:return [[] for _ in questions],status,False
        embedded=self.provider.embed(questions,'query')
        import numpy as np
        query_vectors=np.stack([np.frombuffer(vector,dtype='<f4') for vector in embedded['vectors']])
        with self.connect() as conn:
            for doc_id in ready:
                cursor=conn.execute('''SELECT payload_json,vector FROM semantic_chunks
                    WHERE doc_id=? AND model_key=? AND fingerprint=?''',(doc_id,self.provider.model_key,fingerprints[doc_id]))
                # Bound memory while searching only the selected corpus.
                while batch:=cursor.fetchmany(512):
                    vectors=np.stack([np.frombuffer(r['vector'],dtype='<f4') for r in batch])
                    scores=vectors@query_vectors.T
                    payloads=[json.loads(row['payload_json']) for row in batch]
                    languages=[document_languages[doc_id]]*len(payloads)
                    for column in range(len(questions)):
                        # Languages have different cosine score distributions. Limit
                        # each lane independently before combining; Japanese must not
                        # evict all English hits at the global candidate cutoff.
                        for rank in np.argsort(-scores[:,column])[:limit]:
                            item=dict(payloads[int(rank)])
                            item['semantic_score']=float(scores[int(rank),column])
                            item['retrieval_language']=languages[int(rank)]
                            candidates[column][item['retrieval_language']].append(item)
                        for lane in candidates[column]:
                            candidates[column][lane]=sorted(candidates[column][lane],key=lambda c:(-c['semantic_score'],c['chunk_id']))[:limit]
        combined=[]
        for lanes in candidates:
            hits=[]
            for lane in lanes.values():
                for rank,item in enumerate(lane,1):hits.append(dict(item,language_rank=rank))
            combined.append(sorted(hits,key=lambda c:(c['language_rank'],-c['semantic_score'],c['chunk_id'])))
        return combined,status,embedded.get('query_truncated',False)

    def close(self):
        self.stop.set();self.provider.close()
        if self.worker and self.worker.is_alive():self.worker.join(timeout=5)
