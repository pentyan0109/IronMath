"""Stage 1 training: mod-3 checker on all 8-bit integers (Train 200 / Validation 56), seeds 0,1,2."""
import csv, hashlib, json, random, time
import numpy as np
import torch
from torch import nn
from m3_common import HERE, Checker, n_params, int_bits, true_residues, onehot, peak_ram_mib

cfg = json.load(open(HERE / 'config.json', encoding='utf-8'))['stage1']['training']
(HERE / 'checkpoints').mkdir(exist_ok=True); (HERE / 'data').mkdir(exist_ok=True)

values = list(range(256)); random.Random(92000).shuffle(values)
train_v, val_v = values[:200], values[200:]
for v in values:  # Python-int labels vs vectorised labels
    b = int_bits(v, 8); r = true_residues(b)
    assert all(int(r[i]) == (v & ((1 << (i + 1)) - 1)) % 3 for i in range(8))
split = dict(train=train_v, validation=val_v, note='Validation prefixes overlap Train; used only for checkpoint selection, not for generalization')
json.dump(split, open(HERE / 'data' / 'stage1_split.json', 'w'), indent=0)


def tensors(vs):
    bits = np.stack([int_bits(v, 8) for v in vs]); y = np.stack([true_residues(b) for b in bits]).astype(np.int64)
    return onehot(bits), torch.from_numpy(y)


Xv, Yv = tensors(val_v); ce = nn.CrossEntropyLoss()
log = []; summary = {}
for seed in cfg['seeds']:
    torch.manual_seed(seed); m = Checker(); order = random.Random(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    t0 = time.process_time(); w0 = time.perf_counter(); pool = []; best = None

    def validate(step):
        global best
        m.eval()
        with torch.no_grad():
            lv = ce(m(Xv).reshape(-1, 3), Yv.reshape(-1)).item()
            acc = (m(Xv).argmax(-1) == Yv).float().mean().item()
        m.train()
        log.append(dict(seed=seed, update=step, validation_ce=lv, validation_step_acc=acc, train_ce_last=last,
                        cpu_s=round(time.process_time() - t0, 2), wall_s=round(time.perf_counter() - w0, 2)))
        if best is None or lv < best[0]:  # strict: earliest wins ties
            best = (lv, step, {k: v.clone() for k, v in m.state_dict().items()})

    last = float('nan')
    validate(0)
    for step in range(1, cfg['updates'] + 1):
        batch = []
        while len(batch) < cfg['batch']:
            if not pool:
                pool = train_v[:]; order.shuffle(pool)
            batch.append(pool.pop())
        X, Y = tensors(batch)
        loss = ce(m(X).reshape(-1, 3), Y.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), cfg['clip']); opt.step()
        last = loss.item()
        if step % 100 == 0 or step == cfg['updates']:
            validate(step)
    cpu, wall = time.process_time() - t0, time.perf_counter() - w0
    path = HERE / 'checkpoints' / f'checker_s{seed}.pt'
    torch.save(dict(model=best[2], seed=seed, selected_update=best[1], validation_ce=best[0], n_params=n_params(m)), path)
    torch.save(dict(model=m.state_dict(), seed=seed, update=cfg['updates']), HERE / 'checkpoints' / f'checker_s{seed}_final.pt')
    summary[seed] = dict(selected_update=best[1], selected_validation_ce=best[0], n_params=n_params(m),
                         final_gate_tanh=float(torch.tanh(best[2]['residual_gate'])), train_cpu_s=round(cpu, 2), train_wall_s=round(wall, 2),
                         checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    print(seed, summary[seed], flush=True)

with open(HERE / 'training_log.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, list(log[0].keys())); w.writeheader(); w.writerows(log)
summary['peak_ram_mib'] = peak_ram_mib()
json.dump(summary, open(HERE / 'training_summary.json', 'w'), indent=1)
print('peak RAM MiB', summary['peak_ram_mib'])
