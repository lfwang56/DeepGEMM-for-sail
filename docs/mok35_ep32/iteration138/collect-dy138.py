"""Require every rank's mover reference, source hash, exit and host audit."""
import hashlib,json,math,shlex,subprocess
from pathlib import Path

root='/root/lfwang/Projects/t-head/worktrees/DeepGEMM-for-sail-mok-bf16-training-20261120'
program="""import pathlib,json,sys
r=pathlib.Path(sys.argv[1])/'build/campaign';files={}
for d in r.glob('mok32-opt138-dy-ilp*'):
 if d.is_dir():
  for f in d.glob('rank*.log'):files[str(f.relative_to(r))]=f.read_text(errors='replace')
for f in r.glob('launch-mok32-opt138-dy-ilp*-node*.log'):files[str(f.relative_to(r))]=f.read_text(errors='replace')
print(json.dumps(files))
"""
raw=subprocess.check_output(['ssh','-T','-J','root@8.130.51.79','root@172.17.0.218',shlex.join(['python3','-c',program,root])],text=True)
files=json.loads(raw);out=Path('outputs/mok-qwen35-ep32-opt-20261123');phases=[]
for ilp in (1,2,4,8):
 name=f'mok32-opt138-dy-ilp{ilp}';folder=out/(name+'-logs');folder.mkdir(exist_ok=True)
 expected=hashlib.sha256(Path('work/implementation/deep_gemm/include/deep_gemm/impls/peer_moe_training.cuh').read_bytes()).hexdigest() if ilp==1 else hashlib.sha256(Path(f'work/next138/ilp{ilp}/deep_gemm/include/deep_gemm/impls/peer_moe_training.cuh').read_bytes()).hexdigest()
 ranks=[];audits=[];metrics=[]
 for file,content in sorted(files.items()):
  if not (file.startswith(name+'/') or file.startswith('launch-'+name+'-')):continue
  (folder/Path(file).name).write_text(content);records=[]
  for line in content.splitlines():
   try:r=json.loads(line)
   except ValueError:continue
   if isinstance(r,dict):records.append(r)
  if Path(file).name.startswith('rank'):
   refs=[r for r in records if r.get('case')=='dy138_independent_reference']
   done=[r for r in records if r.get('case')=='dy138_finished']
   src=[r for r in records if r.get('case')=='dy138_source']
   lib=[r for r in records if r.get('case')=='verified_isolated_library']
   valid=(len(refs)==4 and {r['ctas_per_cu'] for r in refs}=={2,4,8,16} and all(r.get('passed') for r in refs)
    and len(done)==1 and done[0].get('passed') and len(src)==1 and src[0]['header_sha256']==expected
    and len(lib)==1 and lib[0]['extension_sha256']=='6a67d3f81f30365ec1f95edb275ed81dca41ff9f24f2513544147ea1f7e1cd01')
   ranks.append(dict(file=file,clean=bool(valid),source=src,references=refs,finished=done))
   metrics.extend(r for r in records if r.get('case')=='dy138_benchmark')
  else:
   pre=[r for r in records if r.get('case')=='idle_node_gate'];post=[r for r in records if r.get('case')=='idle_node_postflight'];parent=[r for r in records if r.get('case')=='local_ranks_finished']
   valid=(len(pre)==1 and pre[0].get('allowed') and len(post)==1 and post[0].get('allowed')
    and post[0].get('command_returncode')==0 and not post[0].get('monitoring_errors') and not post[0].get('observed_foreign_during_command')
    and len(parent)==1 and parent[0].get('passed') and parent[0].get('return_codes')==[0]*8)
   audits.append(dict(file=file,clean=bool(valid),records=records))
 passed=(len(ranks)==8 and len(audits)==1 and all(r['clean'] for r in ranks+audits) and len(metrics)==4
  and all(r.get('passed') and math.isfinite(r['median_max_rank_event_ms']) and r['median_max_rank_event_ms']>0 for r in metrics))
 phases.append(dict(name=name,ilp=ilp,passed=passed,scope='Mover only; 8 GPUs on one host, not EP32 MoE timing',metrics=metrics,ranks=ranks,audits=audits))
 print(name,passed,[(r['ctas_per_cu'],round(r['median_max_rank_event_ms'],5)) for r in metrics])
(out/'dy138-results.json').write_text(json.dumps(dict(passed=all(p['passed'] for p in phases),phases=phases),indent=2)+'\n')
