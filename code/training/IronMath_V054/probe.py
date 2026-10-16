import json,time
import torch
from model import Residual,ternary_length
from run import R,memory,setup
setup(0);rows=[]
for k in ('B0','T1'):
    m=torch.jit.script(Residual(k));x=torch.zeros(16,9 if k=='B0' else ternary_length(8),3)
    for _ in range(3):m.zero_grad();m(x).square().mean().backward()
    c=time.process_time();w=time.perf_counter()
    for _ in range(10):m.zero_grad();m(x).square().mean().backward()
    bc=(time.process_time()-c)/10;bw=(time.perf_counter()-w)/10
    with torch.no_grad():
        n=2049 if k=='B0' else ternary_length(2048);c=time.process_time();w=time.perf_counter();m(torch.zeros(128,n,3));ic=time.process_time()-c;iw=time.perf_counter()-w
    rows.append(dict(model=k,dry_backward_CPU_per_update=bc,dry_backward_wall_per_update=bw,maximum_sequence_length=n,batch128_inference_CPU=ic,batch128_inference_wall=iw,peak_RAM_bytes=memory()))
result=dict(optimizer_updates=0,real_labels_used=False,runs=24,probe=rows,estimated_training_CPU_seconds=sum(r['dry_backward_CPU_per_update']*1200*6+90*6 for r in rows),estimated_all_width_inference_wall_seconds=sum(r['batch128_inference_wall']*4*2*12*2 for r in rows),note='dummy resource probe; optimizer/validation/logging add time; factor2 width sum conservative')
(R/'runtime_estimate.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
