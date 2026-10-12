"""Server-only Gemini settings with Windows per-user DPAPI key protection."""
import base64
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import re
import threading
import uuid

from fastapi import HTTPException

DEFAULT_MODEL='gemini-3.8-flash'


def protect(data,decode=False):
    if os.name!='nt':raise RuntimeError('APIキーの保護にはWindowsが必要です。')
    class Blob(ctypes.Structure):
        _fields_=[('size',wintypes.DWORD),('data',ctypes.POINTER(ctypes.c_ubyte))]
    buffer=ctypes.create_string_buffer(data)
    source=Blob(len(data),ctypes.cast(buffer,ctypes.POINTER(ctypes.c_ubyte)));target=Blob()
    crypt=ctypes.WinDLL('crypt32',use_last_error=True)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.LocalFree.argtypes=[ctypes.c_void_p];kernel.LocalFree.restype=ctypes.c_void_p
    if decode:
        function=crypt.CryptUnprotectData
        function.argtypes=[ctypes.POINTER(Blob),ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(Blob)]
        ok=function(ctypes.byref(source),None,None,None,None,1,ctypes.byref(target))
    else:
        function=crypt.CryptProtectData
        function.argtypes=[ctypes.POINTER(Blob),wintypes.LPCWSTR,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(Blob)]
        ok=function(ctypes.byref(source),'MyNotebookLM research',None,None,None,1,ctypes.byref(target))
    if not ok:raise RuntimeError('APIキーを保護・読込できません。設定で登録し直してください。')
    try:return ctypes.string_at(target.data,target.size)
    finally:kernel.LocalFree(ctypes.cast(target.data,ctypes.c_void_p))


class AISettings:
    def __init__(self,path):self.path=Path(path);self.lock=threading.Lock()
    def read(self):
        if not self.path.exists():return {'provider':'gemini','model':DEFAULT_MODEL,'enabled':False,'version':'initial'}
        try:return json.loads(self.path.read_text(encoding='utf-8'))
        except (ValueError,OSError):raise HTTPException(503,'AI設定を読み込めません。設定を確認してください。') from None
    def public(self):
        data=self.read()
        has_key=bool(data.get('protected_key'));needs_registration=False
        if has_key:
            try:protect(base64.b64decode(data['protected_key'],validate=True),decode=True)
            except (ValueError,RuntimeError):has_key=False;needs_registration=True
        return {k:data[k] for k in ('provider','model','enabled','version')} | {'has_key':has_key,'key_needs_registration':needs_registration}
    def save(self,model,enabled,key='',forget=False,version=None):
        if not re.fullmatch(r'gemini-[a-zA-Z0-9.-]{1,80}',model):raise HTTPException(400,'Geminiのモデル名を確認してください。')
        key=key.strip()
        # Treat credentials as opaque header values; auth keys can contain dots.
        if len(key)>4096:raise HTTPException(400,'APIキーが長すぎます。キーだけをコピーしてください。')
        if key and not re.fullmatch(r'[\x21-\x7e]+',key):raise HTTPException(400,'APIキーに空白・改行・全角文字が含まれています。キーだけをコピーしてください。')
        with self.lock:
            data=self.read()
            if version is not None and data['version']!=version:raise HTTPException(409,'別の画面でAI設定が変更されています。開き直してください。')
            if forget:data.pop('protected_key',None)
            if key:
                try:data['protected_key']=base64.b64encode(protect(key.encode())).decode()
                except RuntimeError:raise HTTPException(503,'APIキーを保護して保存できませんでした。') from None
            if enabled and not data.get('protected_key'):raise HTTPException(400,'APIキーを入力してください。')
            if enabled:
                try:protect(base64.b64decode(data['protected_key'],validate=True),decode=True)
                except (ValueError,RuntimeError):raise HTTPException(400,'このPCでは保存済みAPIキーを読み込めません。APIキーを登録し直してください。') from None
            data.update(provider='gemini',model=model,enabled=enabled,version=str(uuid.uuid4()))
            self.path.parent.mkdir(parents=True,exist_ok=True)
            temp=self.path.with_suffix('.tmp')
            temp.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8');temp.replace(self.path)
            return self.public()
    def credentials(self):
        data=self.read()
        if not data['enabled'] or not data.get('protected_key'):raise HTTPException(400,'AI設定でAPIキーとGeminiの利用を設定してください。')
        try:key=protect(base64.b64decode(data['protected_key']),decode=True).decode()
        except (ValueError,RuntimeError):raise HTTPException(503,'APIキーを読み込めません。AI設定で登録し直してください。') from None
        return data,key
