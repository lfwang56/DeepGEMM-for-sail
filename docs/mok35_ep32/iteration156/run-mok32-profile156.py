import json,subprocess,shlex
from pathlib import Path
r='/root/lfwang/Projects/t-head/worktrees/DeepGEMM-for-sail-mok-bf16-training-20261120'
cs=json.loads(Path('work/mok32-opt156-profile-cases.json').read_text())
p=subprocess.run(['python','work/run-distributed-matrix.py','--nodes','172.17.0.225,172.17.0.226,172.17.0.227,172.17.0.228','--cases','work/mok32-opt156-profile-cases.json','--port','0','--container','wlf_sgl_2.2'])
if p.returncode:raise SystemExit(p.returncode)
for c in cs:
 base=r+'/build/campaign/'+c['name']
 cmd=shlex.join(['docker','exec','wlf_sgl_2.2','asys','export','--force-overwrite=true','--output',base+'.sqlite',base+'.asysrep'])
 subprocess.run(['ssh','-T','-J','root@8.130.51.79','root@172.17.0.225',cmd],check=True)
 subprocess.run(['scp','-q','-o','ProxyJump=root@8.130.51.79','root@172.17.0.225:'+base+'.sqlite','outputs/mok-qwen35-ep32-opt-20261123/'+c['name']+'.sqlite'],check=True)
 print(json.dumps(dict(case='profile156_exported',phase=c['name'])),flush=True)
