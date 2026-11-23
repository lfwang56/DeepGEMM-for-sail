import json,subprocess
from pathlib import Path
for cases in json.loads(Path('work/mok32-opt151-compute-groups.json').read_text()):
 r=subprocess.run(['python','work/run-distributed-matrix.py','--nodes','172.17.0.80','--cases',cases,'--port','0','--container','wlf_sgl_2.2'])
 print(json.dumps(dict(case='compute_geometry_group_finished',cases=cases,returncode=r.returncode)),flush=True)
 if r.returncode==75:raise SystemExit(75)
