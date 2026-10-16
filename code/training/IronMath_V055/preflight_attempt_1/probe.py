import json,time
import torch
from core import Residual
from utils import *
from data import prepare,identity
setup(0);source,data,excluded=prepare();counts={k:len(v) for k,v in data.items()};dump('split_preview.json',dict(counts=counts,exclusions=len(excluded),before_any_Test_predictions=True));m=torch.jit.script(Residual());x=torch.zeros(16,9,3)
for _ in range(3):m.zero_grad();m(x).square().mean().backward()
c=time.process_time();w=time.perf_counter()
for _ in range(10):m.zero_grad();m(x).square().mean().backward()
bc=(time.process_time()-c)/10;bw=(time.perf_counter()-w)/10
with torch.no_grad():
    c=time.process_time();w=time.perf_counter();m(torch.zeros(128,2049,3));ic=time.process_time()-c;iw=time.perf_counter()-w
result=dict(optimizer_updates=0,real_labels_used=False,parameters=1000334,dry_backward_CPU_per_update=bc,dry_backward_wall_per_update=bw,batch128_width2049_inference_CPU=ic,batch128_width2049_inference_wall=iw,peak_RAM_bytes=memory(),new_training_runs=18,estimated_learning_CPU_seconds=9*1200*bc+9*90,estimated_fixed_inference_wall_seconds=iw*4*2*(18*2+6),note='all-width factor2, 18Fbothops+6Z1SUB; source reference inference is cached, not rerun; optimizer/Validation/serialization/diagnostics add cost')
dump('runtime_estimate.json',result);print(json.dumps(result),flush=True)
