"""Install an isolated CPU runtime and pinned, hash-verified public model files."""
from pathlib import Path
import hashlib
import json
import subprocess
import urllib.request
from repair_runtime import ensure_venv

from app.paths import APP_PATHS
APP=APP_PATHS.root
RUNTIME=APP_PATHS.research_runtime
REVISION='614241f622f53c4eeff9890bdc4f31cfecc418b3'
MODEL=APP_PATHS.research_model(REVISION)
FILES=['config.json','tokenizer.json','tokenizer_config.json','onnx/model_qint8_avx512_vnni.onnx']

def main():
    python=ensure_venv(RUNTIME)
    subprocess.run([str(python),'-m','pip','install','--only-binary=:all:','--disable-pip-version-check','-r',str(APP/'requirements-research.txt')],check=True)
    request=urllib.request.Request('https://huggingface.co/api/models/intfloat/multilingual-e5-small/revision/'+REVISION+'?blobs=true',headers={'User-Agent':'MyNotebookLM-local-research-setup'})
    with urllib.request.urlopen(request,timeout=30) as response:
        info=json.load(response)
    assert info['sha']==REVISION
    entries={item['rfilename']:item for item in info['siblings']}
    MODEL.mkdir(parents=True,exist_ok=True)
    manifest={'model':'intfloat/multilingual-e5-small','revision':REVISION,'license':'mit','files':{},'python':python.relative_to(APP).as_posix()}
    for name in FILES:
        entry=entries[name];target=MODEL/name;target.parent.mkdir(parents=True,exist_ok=True)
        def valid(path):
            if not path.is_file():return False
            data=path.read_bytes()
            if entry.get('lfs'):
                return len(data)==entry['lfs']['size'] and hashlib.sha256(data).hexdigest()==entry['lfs']['sha256']
            return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==entry['blobId']
        if not valid(target):
            temporary=target.with_name(target.name+'.download')
            url='https://huggingface.co/intfloat/multilingual-e5-small/resolve/'+REVISION+'/'+name
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'MyNotebookLM-local-research-setup'}),timeout=45) as response,temporary.open('wb') as output:
                while chunk:=response.read(1024*1024):output.write(chunk)
            if not valid(temporary):raise RuntimeError('Model hash validation failed: '+name)
            temporary.replace(target)
        manifest['files'][name]={'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'size':target.stat().st_size}
        print('Verified',name,target.stat().st_size,flush=True)
    versions=subprocess.check_output([str(python),'-c','import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in ["numpy","tokenizers","onnxruntime"]}))'],text=True)
    manifest['versions']=json.loads(versions)
    (MODEL/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (APP_PATHS.models/'research/runtime.json').write_text(json.dumps({'model_path':MODEL.relative_to(APP).as_posix(),**manifest},indent=2),encoding='utf-8')
    print('Isolated local semantic runtime ready.',flush=True)


if __name__=='__main__':
    main()
