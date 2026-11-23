import hashlib,json,sys,runpy
from pathlib import Path
library=Path(sys.argv[sys.argv.index('--library')+1])
header=library/'deep_gemm/include/deep_gemm/impls/peer_moe_training.cuh'
print(json.dumps(dict(case='verified_gradient_header',library=str(library),header_sha256=hashlib.sha256(header.read_bytes()).hexdigest())),flush=True)
runpy.run_path(str(Path(__file__).with_name('run-with-library.py')),run_name='__main__')
