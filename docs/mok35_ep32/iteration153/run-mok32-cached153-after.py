import time,json,subprocess
from pathlib import Path
while True:
 running=False
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:parts=p.read_bytes().split(b'\0')
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if b'work/run-distributed-matrix.py' in parts and b'work/mok32-opt152-sourceq-cases.json' in parts:running=True
 if not running:break
 time.sleep(3)
records=[json.loads(x) for x in Path('work/mok32-opt152-sourceq-controller.log').read_text().splitlines() if x.startswith('{')]
phases=[r for r in records if r.get('case')=='distributed_matrix_phase']
assert len(phases)==10 and all(r['returncode']==0 for r in phases),'source152 prerequisite incomplete or failed'
r=subprocess.run(['python','work/run-distributed-matrix.py','--nodes','172.17.0.88','--cases','work/mok32-opt153-cached-cases.json','--port','0','--container','wlf_sgl_2.2'])
raise SystemExit(r.returncode)
