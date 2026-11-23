import hashlib,json,runpy,sys
from pathlib import Path
library=Path(sys.argv[sys.argv.index('--library')+1])
p=library/'deep_gemm/training/peer.py'
print(json.dumps(dict(case='verified_sourceq_python',library=str(library),peer_py_sha256=hashlib.sha256(p.read_bytes()).hexdigest())),flush=True)
runpy.run_path(str(Path(__file__).with_name('run-with-gradient-library.py')),run_name='__main__')
