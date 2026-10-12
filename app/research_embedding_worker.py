"""Offline ONNX CPU worker. JSON-lines only on stdout; no source text networking."""
import base64
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class Encoder:
    def __init__(self, folder):
        folder=Path(folder)
        self.tokenizer=Tokenizer.from_file(str(folder/'tokenizer.json'))
        options=ort.SessionOptions()
        options.intra_op_num_threads=2;options.inter_op_num_threads=1
        options.log_severity_level=3
        self.session=ort.InferenceSession(str(folder/'onnx/model_qint8_avx512_vnni.onnx'),
                                         sess_options=options,providers=['CPUExecutionProvider'])

    def windows(self, text, prefix):
        # Preserve original character offsets, including punctuation and whitespace.
        offsets=self.tokenizer.encode(text,add_special_tokens=False).offsets
        if not offsets:return []
        budget=min(384,500-len(self.tokenizer.encode(prefix).ids))
        if budget<64:raise ValueError('Passage context is too long')
        windows=[];token_start=0
        while token_start<len(offsets):
            token_end=min(len(offsets),token_start+budget)
            start=0 if token_start==0 else offsets[token_start][0]
            end=len(text) if token_end==len(offsets) else offsets[token_end][0]
            end=min(end,start+1000)
            while end>start and len(self.tokenizer.encode(prefix+text[start:end]).ids)>512:end-=1
            if end<=start:raise ValueError('Cannot create a token-bounded passage')
            if windows and start<=windows[-1][0]:raise ValueError('Passage offsets did not advance')
            windows.append([start,end])
            if end==len(text):break
            covered=token_start
            while covered<len(offsets) and offsets[covered][0]<end:covered+=1
            token_start=max(token_start+1,covered-64)
        return windows

    def embed(self, texts, kind):
        prefix='query: ' if kind=='query' else ''
        encoded=self.tokenizer.encode_batch([prefix+text for text in texts])
        truncated=any(len(item.ids)>512 for item in encoded)
        if truncated and kind!='query':raise ValueError('A passage exceeds the model token budget')
        if truncated:
            self.tokenizer.enable_truncation(max_length=512)
            try:encoded=self.tokenizer.encode_batch([prefix+text for text in texts])
            finally:self.tokenizer.no_truncation()
        length=min(512,max(len(item.ids) for item in encoded))
        ids=np.full((len(texts),length),1,dtype=np.int64)
        mask=np.zeros_like(ids);types=np.zeros_like(ids)
        for row,item in enumerate(encoded):
            n=min(length,len(item.ids));ids[row,:n]=item.ids[:n];mask[row,:n]=1;types[row,:n]=item.type_ids[:n]
        available={'input_ids':ids,'attention_mask':mask,'token_type_ids':types}
        feeds={item.name:available[item.name] for item in self.session.get_inputs()}
        hidden=self.session.run(None,feeds)[0]
        weights=mask[...,None].astype(np.float32)
        embeddings=(hidden.astype(np.float32)*weights).sum(axis=1)/weights.sum(axis=1).clip(1)
        embeddings/=np.linalg.norm(embeddings,axis=1,keepdims=True).clip(1e-12)
        if embeddings.shape!=(len(texts),384) or not np.isfinite(embeddings).all():raise ValueError('Invalid embedding dimensions')
        return {'vectors':[base64.b64encode(row.astype('<f4').tobytes()).decode('ascii') for row in embeddings],
                'query_truncated':truncated}


def main():
    sys.stdin.reconfigure(encoding='utf-8');sys.stdout.reconfigure(encoding='utf-8')
    encoder=Encoder(sys.argv[1])
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if request['op']=='windows':result={'windows':encoder.windows(request['text'],request['prefix'])}
            elif request['op']=='embed':result=encoder.embed(request['texts'],request['kind'])
            elif request['op']=='ping':result={'dimensions':384,'provider':'local-onnx-cpu'}
            else:raise ValueError('Unknown operation')
            print(json.dumps({'ok':True,**result},ensure_ascii=False),flush=True)
        except Exception as error:
            print(json.dumps({'ok':False,'error':str(error)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
