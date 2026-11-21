"""Export a measured, single-rank graph timeline without equating mixed time to wire time."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

p=argparse.ArgumentParser()
p.add_argument('--serial',required=True)
p.add_argument('--parallel',required=True)
p.add_argument('--output',required=True)
p.add_argument('--serial-title',default='Serial weight gradients')
p.add_argument('--parallel-title',default='Concurrent weight gradients')
a=p.parse_args()
colors={'Weight gradients':'#E69F00','Local GEMMs':'#009E73',
        'Mixed GEMM / peer traffic':'#0072B2','Communication / score work':'#CC79A7',
        'GEMM with input-ready waits':'#56B4E9',
        'Rank waits':'#D55E00','Routing / copies / zeroing':'#999999'}

def category(name):
    if name.startswith('bf16_k_grouped_'):return 'Weight gradients'
    if name.startswith(('bf16_peer_dag_', 'bf16_peer_ffn_')):return 'Mixed GEMM / peer traffic'
    if name.startswith('bf16_peer_'):
        if '_external_dispatch' in name:return 'GEMM with input-ready waits'
        return 'Mixed GEMM / peer traffic' if '_rtrue_' in name or '_wtrue_' in name or '_pipeline' in name or '_replay_grad' in name else 'Local GEMMs'
    if name=='moe_training_peer_5':return 'Rank waits'
    if name in ('moe_training_peer_13','moe_training_peer_15','moe_training_peer_16','moe_training_peer_17') or 'deep_ep::' in name:
        return 'Communication / score work'
    if name.startswith('bf16_grouped_'):return 'Local GEMMs'
    return 'Routing / copies / zeroing'

def load(path):
    c=sqlite3.connect(path)
    rows=list(c.execute('''SELECT k.start,k.end,s.value,k.streamId,k.correlationId
        FROM HGPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start'''))
    grouped=defaultdict(list)
    for row in rows:grouped[row[4]].append(row)
    graphs=sorted((g for g in grouped.values() if len(g)>20),key=lambda g:min(r[0] for r in g))
    assert len(graphs)==3,(path,[len(g) for g in graphs])
    graph=graphs[1]
    origin=min(r[0] for r in graph)
    main=min(r[3] for r in graph)
    converted=[((begin-origin)/1e6,(end-begin)/1e6,category(name),0 if stream==main else 1,name)
               for begin,end,name,stream,_ in graph]
    return converted,dict(source=str(path),correlation_id=graph[0][4],kernels=len(graph),
        span_ms=(max(r[1] for r in graph)-origin)/1e6,streams=sorted({r[3] for r in graph}))

fig,axes=plt.subplots(2,1,figsize=(13,5.3),sharex=True)
metadata=[]
used_categories=set()
for axis,path,title in zip(axes,[a.serial,a.parallel],[a.serial_title,a.parallel_title]):
    rows,meta=load(path);metadata.append(meta)
    used_categories.update(row[2] for row in rows)
    for start,width,kind,lane,_ in rows:
        axis.broken_barh([(start,width)],(1-lane-.31,.62),facecolors=colors[kind],edgecolors='none')
    axis.set_yticks([1,0],[f'Device stream {meta["streams"][0]}', f'Device stream {meta["streams"][1]}' if len(meta['streams']) > 1 else 'No second stream'])
    axis.tick_params(axis='y',labelsize=9)
    axis.set_ylim(-.65,1.65)
    axis.set_title(f'{title}: sampled rank0 graph span {meta["span_ms"]:.2f} ms',loc='left',fontsize=11)
    axis.grid(axis='x',alpha=.2)
    axis.spines[['top','right']].set_visible(False)
    axis.axvline(meta['span_ms'],color='#333333',linewidth=.8,linestyle='--')
axes[-1].set_xlabel('Time from the first kernel of the middle graph replay (ms)')
axes[-1].set_xlim(0,max(x['span_ms'] for x in metadata)*1.02)
fig.suptitle('Qwen routed-layer dimensions, domain EP32 — measured GPU kernel intervals',fontsize=13)
fig.legend(handles=[Patch(color=v,label=k) for k,v in colors.items() if k in used_categories],loc='lower center',ncol=3,
           bbox_to_anchor=(.5,.035),fontsize=9,frameon=False)
fig.text(.02,.012,'Rank0 Asys kernel spans include waits. Mixed kernels include peer traffic; overlaps do not isolate wire or active MMA time.',fontsize=8)
fig.subplots_adjust(left=.16,right=.985,top=.87,bottom=.24,hspace=.55)
base=Path(a.output);base.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(str(base)+'.png',dpi=200,bbox_inches='tight')
fig.savefig(str(base)+'.svg',bbox_inches='tight')
base.with_suffix('.json').write_text(json.dumps(dict(graphs=metadata,
    scope='Single-rank instrumented task intervals; use unprofiled maximum-rank medians for performance'),indent=2)+'\n')
print(json.dumps(metadata))
