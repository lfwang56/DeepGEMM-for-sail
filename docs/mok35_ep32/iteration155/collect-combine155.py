"""Validate every component rank and require bitwise identity to the old kernel."""
from pathlib import Path
import hashlib,json,math,shlex,subprocess
root='/root/lfwang/Projects/t-head/worktrees/DeepGEMM-for-sail-mok-bf16-training-20261120'
program="""from pathlib import Path
import json,sys
r=Path(sys.argv[1])/'build/campaign';files={}
for d in r.glob('mok32-opt155r2-combine-*'):
 if d.is_dir():
  for f in d.glob('rank*.log'):files[str(f.relative_to(r))]=f.read_text(errors='replace')
for f in r.glob('launch-mok32-opt155r2-combine-*-node*.log'):files[str(f.relative_to(r))]=f.read_text(errors='replace')
print(json.dumps(files))
"""
raw=subprocess.check_output(['ssh','-T','-J','root@8.130.51.79','root@172.17.0.218',shlex.join(['python3','-c',program,root])],text=True)
files=json.loads(raw);out=Path('outputs/mok-qwen35-ep32-opt-20261123');phases=[]
for label,sha in [('baseline','6a67d3f81f30365ec1f95edb275ed81dca41ff9f24f2513544147ea1f7e1cd01'),('vector','ef2ba3282c261c5781bac993126e82ea0b0b9a4a2f00787ce1136bd0378ec658')]:
 name='mok32-opt155r2-combine-'+label;logs=out/(name+'-logs');logs.mkdir(exist_ok=True);ranks=[];audits=[]
 for f,s in sorted(files.items()):
  if not(f.startswith(name+'/') or f.startswith('launch-'+name+'-')):continue
  (logs/Path(f).name).write_text(s);records=[]
  for line in s.splitlines():
   try:r=json.loads(line)
   except ValueError:continue
   if isinstance(r,dict):records.append(r)
  if Path(f).name.startswith('rank'):
   refs=[r for r in records if r.get('case')=='combine155_reference'];times=[r for r in records if r.get('case')=='combine155_benchmark'];finish=[r for r in records if r.get('case')=='combine155_finished'];libs=[r for r in records if r.get('case')=='verified_isolated_library']
   passed=(len(refs)==30 and all(r.get('passed') and r['independent_fp64'] for r in refs) and len(times)==4 and all(r.get('passed') and len(r['samples_event_ms'])==5 and len(r['output_hashes'])==5 and all(math.isfinite(t) and t>0 for t in r['samples_event_ms']) for r in times) and len(finish)==1 and finish[0].get('passed') and len(libs)==1 and libs[0]['extension_sha256']==sha and libs[0]['torch']=='2.11.0')
   ranks.append(dict(file=f,passed=bool(passed),library=libs,references=refs,timings=times))
  else:
   pre=[r for r in records if r.get('case')=='idle_node_gate'];post=[r for r in records if r.get('case')=='idle_node_postflight'];parent=[r for r in records if r.get('case')=='local_ranks_finished']
   passed=(len(pre)==1 and pre[0].get('allowed') and len(post)==1 and post[0].get('allowed') and post[0].get('command_returncode')==0 and not post[0].get('observed_foreign_during_command') and not post[0].get('monitoring_errors') and len(parent)==1 and parent[0].get('passed') and parent[0].get('return_codes')==[0]*8)
   audits.append(dict(file=f,passed=bool(passed),records=records))
 phases.append(dict(label=label,passed=len(ranks)==8 and len(audits)==1 and all(r['passed'] for r in ranks+audits),ranks=ranks,audits=audits))
assert all(p['passed'] for p in phases),[(p['label'],p['passed']) for p in phases]
keys=['c','h','k','t','start','offset','weighted','replay'];checks=0
for old,new in zip(phases[0]['ranks'],phases[1]['ranks']):
 assert Path(old['file']).name==Path(new['file']).name
 a={tuple(r[k] for k in keys):r for r in old['references']};b={tuple(r[k] for k in keys):r for r in new['references']};assert a.keys()==b.keys()
 for k in a:assert a[k]['output_sha256']==b[k]['output_sha256'],('reference bits',old['file'],k);checks+=1
 a={(r['tokens'],r['weighted']):r for r in old['timings']};b={(r['tokens'],r['weighted']):r for r in new['timings']};assert a.keys()==b.keys()
 for k in a:assert a[k]['output_hashes']==b[k]['output_hashes'],('performance bits',old['file'],k);checks+=5
summary=[]
for tokens in [2048,4096]:
 for weighted in [False,True]:
  row=dict(tokens=tokens,weighted=weighted)
  for phase in phases:
   times=[next(t for t in r['timings'] if (t['tokens'],t['weighted'])==(tokens,weighted))['samples_event_ms'] for r in phase['ranks']]
   maxima=[max(t[i] for t in times) for i in range(5)];row[phase['label']+'_median_max_rank_event_ms']=sorted(maxima)[2]
  row['speedup']=row['baseline_median_max_rank_event_ms']/row['vector_median_max_rank_event_ms'];summary.append(row);print(row)
result=dict(passed=True,scope='Local combine only, not complete MoE timing',independent_reference_records=480,exact_output_hash_checks=checks,summary=summary,phases=phases)
(out/'combine155-results.json').write_text(json.dumps(result,indent=2)+'\n')
