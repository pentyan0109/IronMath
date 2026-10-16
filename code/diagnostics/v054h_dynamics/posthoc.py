"""事後の補助分析（B の結果を見た後に追加。事前登録外）。
同じ入力対が両方の状態の保持組に現れるもの（ADD (0,1),(1,0) / SUB (0,0),(1,1)）について、
状態0側・状態1側それぞれの保持開始hiddenから同じ固定入力で反復した終点 h* の距離を測る。
あわせて tanh(residual_gate) を記録する。usage: python -B posthoc.py"""
import json, sys
from pathlib import Path
H = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(H))
import torch
from v054h import SEEDS, V054T, bits, csvout, jl_read, load_eager, make_x, step, step_fn

torch.set_num_threads(2)
ITERS = json.loads((H / 'estimate.json').read_text(encoding='utf-8'))['B_fixed_point_iterations']
probes = {(r['op'], r['state'], r['pair']): r for r in jl_read(V054T / 'data' / 'probes.jsonl') if r['k'] == 4096 and r['prefix'] == 'min' and r['suffix'] == 's0'}
rows = []
for s in SEEDS:
    base = load_eager(s); gate = float(torch.tanh(base.residual_gate))
    for op, pair in [('ADD', '01'), ('ADD', '10'), ('SUB', '00'), ('SUB', '11')]:
        ends = {}
        for st in (0, 1):
            r = probes[(op, st, pair)]; x = torch.from_numpy(make_x(op, int(r['a'], 16), int(r['b'], 16), r['seq_len'])); h = torch.zeros(1, 545)
            with torch.no_grad():
                for t in range(r['hold_start']):
                    h, _ = step_fn(base, x[t:t + 1], h)
                xin = x[r['hold_start']:r['hold_start'] + 1]
                for _ in range(ITERS):
                    h, z = step_fn(base, xin, h)
            ends[st] = (h, int(z.argmax(-1)))
        rows.append(dict(seed=s, op=op, pair=pair, tanh_residual_gate=gate, iterations=ITERS,
                         distance_hstar_state0_vs_state1=float((ends[0][0] - ends[1][0]).norm()), hstar_norm_state0=float(ends[0][0].norm()),
                         output_state0_side=ends[0][1], correct_output_state0=step(op, 0, int(pair[0]), int(pair[1]))[0],
                         output_state1_side=ends[1][1], correct_output_state1=step(op, 1, int(pair[0]), int(pair[1]))[0]))
        print(rows[-1], flush=True)
csvout('posthoc_fixed_point_pairs.csv', rows)
