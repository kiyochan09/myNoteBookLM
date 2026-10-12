"""Provider boundary and lifecycle for the isolated, offline embedding runtime."""
import atexit
import base64
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
from typing import Protocol
from app.runtime_paths import get_research_python, python_available

REVISION='614241f622f53c4eeff9890bdc4f31cfecc418b3'
MODEL_KEY='multilingual-e5-small:'+REVISION+':qint8:mean-l2:tokens384-overlap64:context80:v1'


class EmbeddingProvider(Protocol):
    model_key: str
    def windows(self,text:str,prefix:str)->list: ...
    def embed(self,texts:list,kind:str)->dict: ...
    def close(self)->None: ...


class OfflineEmbeddingProvider:
    model_key=MODEL_KEY
    def __init__(self,app_path):
        self.app_path=Path(app_path).resolve()
        self.python=get_research_python(self.app_path)
        self.model=self.app_path/'models/research'/('multilingual-e5-small-'+REVISION[:12])
        self.worker=self.app_path/'app/research_embedding_worker.py'
        self.lock=threading.Lock();self.process=None;self.output=None;self.closed=False
        atexit.register(self.close)

    def available(self):
        self.python=get_research_python(self.app_path)
        return python_available(self.python, ('numpy','tokenizers','onnxruntime')) and (self.model/'manifest.json').is_file() and self.worker.is_file()

    def start(self):
        if self.closed:raise RuntimeError('意味検索を終了しています。画面を再読み込みしてください。')
        if self.process and self.process.poll() is None:return
        if not self.available():raise RuntimeError('意味検索用のモデルまたは実行環境を利用できません。PC移行後は移行手順に沿って実行環境を修復してください。')
        self.output=queue.Queue();output=self.output
        flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        self.process=subprocess.Popen([str(self.python),'-u',str(self.worker),str(self.model)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,
            encoding='utf-8',bufsize=1,creationflags=flags,cwd=self.app_path,env={**os.environ,'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
        process=self.process
        def read():
            try:
                for line in process.stdout:output.put(line)
            finally:output.put(None)
        threading.Thread(target=read,name='research-embedding-output',daemon=True).start()

    def request(self,payload):
        if not self.lock.acquire(timeout=45):raise RuntimeError('意味検索の処理が混み合っています。再試行してください。')
        try:
            self.start()
            self.process.stdin.write(json.dumps(payload,ensure_ascii=False)+'\n');self.process.stdin.flush()
            try:line=self.output.get(timeout=45)
            except queue.Empty:
                self.process.kill();self.process.wait(timeout=5)
                raise RuntimeError('意味検索の処理が時間内に完了しませんでした。再試行してください。')
            if line is None:raise RuntimeError('意味検索の実行部が停止しました。再試行してください。')
            result=json.loads(line)
            if not result.get('ok'):raise RuntimeError('意味検索の処理に失敗しました：'+result.get('error',''))
            return result
        finally:self.lock.release()

    def windows(self,text,prefix):return self.request({'op':'windows','text':text,'prefix':prefix})['windows']

    def embed(self,texts,kind):
        result=self.request({'op':'embed','texts':texts,'kind':kind})
        return {'vectors':[base64.b64decode(v) for v in result['vectors']],
                'query_truncated':result.get('query_truncated',False)}

    def close(self):
        self.closed=True
        process=self.process
        if process and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill()
        self.process=None
