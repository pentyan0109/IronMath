"""Disposable pre-training resource probe; no optimizer update or Test data."""
import json,time
from pathlib import Path
import torch
from model import Residual
torch.set_num_threads(2);torch.manual_seed(0)
m=torch.jit.script(Residual());x=torch.zeros(16,9,3)
for _ in range(3):m.zero_grad();m(x).square().mean().backward()
c=time.process_time();a=time.perf_counter()
for _ in range(20):m.zero_grad();m(x).square().mean().backward()
cpu=(time.process_time()-c)/20;wall=(time.perf_counter()-a)/20
with torch.no_grad():
    m.eval();xx=torch.zeros(128,2049,3);c=time.process_time();a=time.perf_counter();m(xx)
    longcpu=time.process_time()-c;longwall=time.perf_counter()-a
result=dict(before_main_training=True,optimizer_updates=0,parameters=sum(p.numel() for p in m.parameters()),dry_forward_backward_CPU_per_update=cpu,dry_forward_backward_wall_per_update=wall,evaluation_batch128_width2049_CPU=longcpu,evaluation_batch128_width2049_wall=longwall,C1_updates=1200,C2_CPU_seconds=90,total_new_runs=12,estimated_inference_wall_seconds=longwall*4*2*12*2,scope='resource probe on zeros; no real problem labels or learning; full evaluation includes other widths and logging',training_estimate_note='AdamW and Validation add cost beyond dry backward; estimated overall minutes to about1-2hours, no10-hour learning')
Path(__file__).with_name('runtime_estimate.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
