"""Durable answer jobs, bounded context, and mechanically checked source references."""
import asyncio
from contextlib import closing
from hashlib import sha256
import json
import secrets
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from app.research_generation import GeminiProvider, GenerationError, answer_schema
from app.research_threads import ResearchThreads
from app.research_index import resolve_documents

ACTIVE=('queued','generating','validating')


def plain(text):return ''.join(text.split())


def answer_sources(citations):
    selected=[];characters=0
    for source in citations:
        if len(selected)==32:break
        if characters+len(source['quote'])>60000:continue
        selected.append(source);characters+=len(source['quote'])
    return selected


class Proof(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    source_id: str
    quote: str=Field(min_length=8,max_length=500)


class Paragraph(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    heading: str=Field(default='',max_length=120)
    text: str=Field(min_length=1,max_length=3000)
    evidence: list[Proof]=Field(min_length=1,max_length=6)


class Answer(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    status: Literal['answered','insufficient']
    paragraphs: list[Paragraph]=Field(max_length=20)
    limitations: str=Field(max_length=2000)


def validate_answer(raw,sources):
    try:answer=Answer.model_validate(raw)
    except ValueError:raise GenerationError('回答の出典形式を確認できませんでした。再試行してください。') from None
    if (answer.status=='answered')!=bool(answer.paragraphs):raise GenerationError('回答と根拠不足の状態が一致しません。再試行してください。')
    if answer.status=='insufficient' and not answer.limitations.strip():raise GenerationError('根拠不足の説明を確認できませんでした。')
    by_id={s['id']:s for s in sources}
    for paragraph in answer.paragraphs:
        if not paragraph.text.strip():raise GenerationError('空の回答が返されました。')
        for proof in paragraph.evidence:
            source=by_id.get(proof.source_id)
            if not source or len(plain(proof.quote))<8 or plain(proof.quote) not in plain(source['quote']):
                raise GenerationError('引用が検索した原文と一致しません。回答は保存せず、再試行を待っています。')
    return answer.model_dump()


def get_context(store,thread_id,parent_id,tag_ids):
    if not parent_id:return None
    with store.connect() as conn:
        parent=conn.execute("SELECT * FROM research_answers WHERE id=? AND thread_id=? AND status='completed'",(parent_id,thread_id)).fetchone()
        if not parent:raise HTTPException(400,'続きの元になる回答が見つかりません。')
        run=conn.execute('SELECT result_json FROM research_searches WHERE id=? AND thread_id=?',(parent['run_id'],thread_id)).fetchone()
        if not run:raise HTTPException(400,'前の回答の検索結果が見つかりません。')
        run=json.loads(run['result_json'])
        if sorted(run['tagIds'])!=sorted(tag_ids):raise HTTPException(400,'前の回答とはタグが異なります。続きの指定を外して質問してください。')
        answer=json.loads(parent['answer_json'])
    return {'question':run['question'][:1500],
            'answer':'\n'.join(p['text'] for p in answer['paragraphs'])[:2500],
            'limitations':answer['limitations'][:500]}


class AnswerRequest(BaseModel):
    id: str=Field(min_length=1,max_length=100)
    run_id: str=Field(min_length=1,max_length=100)
    revision: int=Field(ge=1)
    parent_id: str|None=Field(default=None,max_length=100)


class SettingsRequest(BaseModel):
    model: str=Field(min_length=1,max_length=90)
    enabled: bool
    api_key: SecretStr=Field(default=SecretStr(''))
    forget_key: bool=False
    version: str=Field(min_length=1,max_length=100)


class ResearchAnswers:
    def __init__(self,path,db_service,settings,provider_factory=None):
        self.store=ResearchThreads(path);self.db_service=db_service;self.settings=settings
        self.factory=provider_factory or (lambda data,key:GeminiProvider(key,data['model']))
        self.tasks={};self.semaphore=asyncio.Semaphore(1)
        with self.store.connect() as conn:
            conn.executescript('''CREATE TABLE IF NOT EXISTS research_answers (
                id TEXT PRIMARY KEY,thread_id TEXT NOT NULL,run_id TEXT NOT NULL,parent_id TEXT,
                status TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,
                provider TEXT NOT NULL,model TEXT NOT NULL,request_json TEXT NOT NULL,
                answer_json TEXT,usage_json TEXT NOT NULL DEFAULT '{}',finish_reason TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '');
                CREATE INDEX IF NOT EXISTS idx_research_answers_thread ON research_answers(thread_id,created_at);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_research_answers_active ON research_answers(thread_id)
                    WHERE status IN ('queued','generating','validating');''')
            conn.execute("UPDATE research_answers SET status='interrupted',error='アプリの再起動で回答作成が中断しました。再試行できます。',updated_at=? WHERE status IN ('queued','generating','validating')",(self.store.now(),))

    def selected(self,conn,thread_id,revision=None):
        row=conn.execute('SELECT * FROM research_threads WHERE id=? AND deleted_at IS NULL',(thread_id,)).fetchone()
        if not row:raise HTTPException(404,'スレッドが見つかりません。')
        if revision is not None and row['revision']!=revision:raise HTTPException(409,'スレッドが変更されています。再読み込みしてください。')
        return self.store.decode(row)

    def run(self,conn,thread_id,run_id):
        row=conn.execute('SELECT result_json FROM research_searches WHERE id=? AND thread_id=?',(run_id,thread_id)).fetchone()
        if not row:raise HTTPException(404,'検索結果が見つかりません。')
        return json.loads(row['result_json'])

    def check_scope(self,selected,run):
        if sorted(selected['tagIds'])!=sorted(run['tagIds']) or sorted(selected['groupIds'])!=sorted(run['groupIds']):
            raise HTTPException(409,'資料のタグが検索時から変わりました。もう一度資料を検索してください。')
        resolve_documents(self.db_service,selected['groupIds'],selected['tagIds'])

    def check_sources(self,sources,tag_ids):
        with closing(self.db_service.get_connection()) as conn:
            conn.execute('PRAGMA query_only=ON');conn.execute('BEGIN')
            marks=','.join('?' for _ in tag_ids)
            docs={r['id']:dict(r) for r in conn.execute(f'''SELECT id,title,updated_at FROM documents
                WHERE id IN (SELECT document_id FROM document_tags WHERE tag_id IN ({marks}))''',tag_ids)}
            for source in sources:
                doc=docs.get(source['document_id'])
                if not doc or doc['updated_at']!=source['updated_at'] or doc['title']!=source['title']:
                    raise HTTPException(409,'資料が検索後に更新されました。もう一度検索してから回答を作成してください。')
                pieces=[]
                for segment in source.get('segments') or [source]:
                    row=conn.execute('SELECT text_content FROM content_blocks WHERE document_id=? AND id=?',(source['document_id'],segment['block_id'])).fetchone()
                    text=row['text_content'] if row else ''
                    if not row or sha256(text.encode()).hexdigest()!=segment['content_hash'] or not 0<=segment['start']<segment['end']<=len(text):
                        raise HTTPException(409,'引用元の本文が変わりました。もう一度検索してください。')
                    pieces.append(text[segment['start']:segment['end']])
                if plain(''.join(pieces))!=plain(source['quote']):raise HTTPException(409,'引用の原文範囲を確認できません。再検索してください。')

    def decode(self,row):
        result={k:row[k] for k in ('id','thread_id','run_id','parent_id','status','created_at','updated_at','provider','model','finish_reason','error')}
        result['answer']=json.loads(row['answer_json']) if row['answer_json'] else None
        result['usage']=json.loads(row['usage_json'])
        result['sent']=json.loads(row['request_json'])['summary']
        return result

    def get(self,thread_id,job_id):
        with self.store.connect() as conn:
            self.selected(conn,thread_id)
            row=conn.execute('SELECT * FROM research_answers WHERE thread_id=? AND id=?',(thread_id,job_id)).fetchone()
            if not row:raise HTTPException(404,'回答が見つかりません。')
            return self.decode(row)

    def list(self,thread_id):
        with self.store.connect() as conn:
            self.selected(conn,thread_id)
            return [self.decode(r) for r in conn.execute('SELECT * FROM research_answers WHERE thread_id=? ORDER BY created_at DESC LIMIT 50',(thread_id,))]

    async def start(self,thread_id,data):
        with self.store.connect() as conn:
            selected=self.selected(conn,thread_id,data.revision)
            existing=conn.execute('SELECT * FROM research_answers WHERE id=?',(data.id,)).fetchone()
            if existing:
                if (existing['thread_id'],existing['run_id'],existing['parent_id'])!=(thread_id,data.run_id,data.parent_id):
                    raise HTTPException(409,'回答IDが重複しています。')
                return self.decode(existing)
            run=self.run(conn,thread_id,data.run_id)
        self.check_scope(selected,run)
        # Bounded contextual excerpts, with exact original offsets and hashes.
        sources=answer_sources(run['citations'])
        if sources:await asyncio.to_thread(self.check_sources,sources,selected['tagIds'])
        context=get_context(self.store,thread_id,data.parent_id,selected['tagIds'])
        if data.parent_id!=run.get('context_answer_id'):
            raise HTTPException(409,'質問の続きの指定が検索時と異なります。もう一度検索してください。')
        config=self.settings.public()
        if sources and (not config['enabled'] or not config['has_key']):
            raise HTTPException(400,'AI設定でAPIキーとGeminiの利用を設定してください。')
        payload={'question':run['question'][:4000],
                 'sources':[{'source_id':s['id'],'title':s['title'][:120],'section':s.get('section_title','')[:120],'quote':s['quote']} for s in sources]}
        if context:payload['previous_turn']=context
        if len(json.dumps(payload,ensure_ascii=False))>72000:raise HTTPException(400,'送信内容が長すぎます。質問を短くしてください。')
        summary={'excerpts':len(sources),'quote_characters':sum(len(s['quote']) for s in sources),
                 'question_truncated':len(run['question'])>4000,'with_context':bool(context)}
        request={'payload':payload,'summary':summary,'settings_version':config['version']}
        now=self.store.now()
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE');self.selected(conn,thread_id,data.revision)
            duplicate=conn.execute('SELECT * FROM research_answers WHERE id=?',(data.id,)).fetchone()
            if duplicate:
                if (duplicate['thread_id'],duplicate['run_id'],duplicate['parent_id'])!=(thread_id,data.run_id,data.parent_id):raise HTTPException(409,'回答IDが重複しています。')
                return self.decode(duplicate)
            if conn.execute("SELECT 1 FROM research_answers WHERE thread_id=? AND status IN ('queued','generating','validating')",(thread_id,)).fetchone():
                raise HTTPException(409,'このスレッドでは回答を作成中です。完了か停止を待ってください。')
            count=conn.execute("SELECT count(*) FROM research_answers WHERE status IN ('queued','generating','validating')").fetchone()[0]
            if count>=3:raise HTTPException(429,'回答の作成が混み合っています。完了を待ってください。')
            conn.execute('INSERT INTO research_answers(id,thread_id,run_id,parent_id,status,created_at,updated_at,provider,model,request_json) VALUES (?,?,?,?,?,?,?,?,?,?)',
                         (data.id,thread_id,data.run_id,data.parent_id,'queued',now,now,config['provider'],config['model'],json.dumps(request,ensure_ascii=False)))
        task=asyncio.create_task(self.work(thread_id,data.id,run,sources,request))
        self.tasks[data.id]=task;task.add_done_callback(lambda _:self.tasks.pop(data.id,None))
        return self.get(thread_id,data.id)

    def update(self,job_id,status,answer=None,usage=None,finish_reason='',error=''):
        with self.store.connect() as conn:
            conn.execute("UPDATE research_answers SET status=?,answer_json=?,usage_json=?,finish_reason=?,error=?,updated_at=? WHERE id=? AND status IN ('queued','generating','validating')",
                         (status,json.dumps(answer,ensure_ascii=False) if answer is not None else None,json.dumps(usage or {}),finish_reason,error,self.store.now(),job_id))

    async def work(self,thread_id,job_id,run,sources,request):
        usage={};reason=''
        try:
            async with self.semaphore:
                if self.get(thread_id,job_id)['status'] not in ACTIVE:return
                with self.store.connect() as conn:selected=self.selected(conn,thread_id)
                self.check_scope(selected,run)
                if not sources:
                    self.update(job_id,'completed',{'status':'insufficient','paragraphs':[],'limitations':'選択した資料に回答の根拠が見つかりませんでした。タグや質問を見直してください。'},finish_reason='NO_EVIDENCE');return
                await asyncio.to_thread(self.check_sources,sources,run['tagIds'])
                config,key=self.settings.credentials()
                if config['version']!=request['settings_version']:raise GenerationError('AI設定が変更されたため作成を停止しました。再試行してください。')
                self.update(job_id,'generating')
                generated=await asyncio.wait_for(self.factory(config,key).generate(request['payload'],answer_schema([s['id'] for s in sources])),timeout=100)
                usage=generated.usage;reason=generated.finish_reason
                self.update(job_id,'validating',usage=usage,finish_reason=reason)
                answer=validate_answer(generated.content,sources)
                with self.store.connect() as conn:selected=self.selected(conn,thread_id)
                self.check_scope(selected,run)
                await asyncio.to_thread(self.check_sources,sources,run['tagIds'])
                self.update(job_id,'completed',answer,usage,reason)
        except asyncio.CancelledError:
            self.update(job_id,'interrupted',usage=usage,finish_reason=reason,error='回答作成を中断しました。再試行できます。')
        except (GenerationError,HTTPException) as error:
            self.update(job_id,'failed',usage=getattr(error,'usage',usage),finish_reason=getattr(error,'finish_reason',reason),
                        error=error.detail if isinstance(error,HTTPException) else str(error))
        except TimeoutError:
            self.update(job_id,'failed',usage=usage,finish_reason=reason,error='回答の作成が時間切れになりました。自動再送はしていません。')
        except Exception:
            self.update(job_id,'failed',usage=usage,finish_reason=reason,error='回答を作成できませんでした。AI設定を確認して再試行してください。')

    async def cancel(self,thread_id,job_id):
        self.get(thread_id,job_id)
        self.update(job_id,'canceled',error='回答の作成を停止しました。送信済みの処理には利用量が発生する場合があります。')
        task=self.tasks.get(job_id)
        if task:task.cancel()
        return self.get(thread_id,job_id)

    async def close(self):
        tasks=list(self.tasks.values())
        for task in tasks:task.cancel()
        if tasks:await asyncio.gather(*tasks,return_exceptions=True)


def create_answers_router(manager,token):
    def authenticate(request:Request):
        if not secrets.compare_digest(request.headers.get('x-app-token',''),token):raise HTTPException(401,'画面を再読み込みしてください。')
    router=APIRouter(prefix='/api/research',dependencies=[Depends(authenticate)])

    @router.get('/ai/settings')
    def settings():return {'settings':manager.settings.public()}

    @router.put('/ai/settings')
    def save_settings(data:SettingsRequest):
        return {'settings':manager.settings.save(data.model,data.enabled,data.api_key.get_secret_value(),data.forget_key,data.version)}

    @router.get('/threads/{thread_id}/answers')
    def answers(thread_id:str):return {'answers':manager.list(thread_id)}

    @router.post('/threads/{thread_id}/answers')
    async def start(thread_id:str,data:AnswerRequest):return {'answer':await manager.start(thread_id,data)}

    @router.get('/threads/{thread_id}/answers/{job_id}')
    def detail(thread_id:str,job_id:str):return {'answer':manager.get(thread_id,job_id)}

    @router.delete('/threads/{thread_id}/answers/{job_id}')
    async def stop(thread_id:str,job_id:str):return {'answer':await manager.cancel(thread_id,job_id)}
    return router
