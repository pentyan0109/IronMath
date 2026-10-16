"""M3 checker: shared model / encoding / inference.
Residual step h = GRUCell(z_t, h) + tanh(residual_gate) * h is taken from IronMath_V4/v051/src/model.py (A1),
with sizes reduced as instructed: Linear(2->32) -> GRUCell(32->64) -> Linear(64->3)."""
import ctypes, ctypes.wintypes, os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)


class Checker(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 32); self.core = nn.GRUCell(32, 64)
        self.residual_gate = nn.Parameter(torch.zeros(())); self.head = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (B,T,2) one-hot -> logits (B,T,3)
        z = self.projection(x); h = torch.zeros(x.size(0), 64, dtype=x.dtype, device=x.device)
        out = torch.jit.annotate(List[torch.Tensor], [])
        for i in range(x.size(1)):
            h = self.core(z[:, i], h) + torch.tanh(self.residual_gate) * h; out.append(self.head(h))
        return torch.stack(out, 1)

    @torch.jit.export
    def predict_chunk(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.projection(x); g = torch.tanh(self.residual_gate)
        out = torch.jit.annotate(List[torch.Tensor], [])
        for i in range(x.size(1)):
            h = self.core(z[:, i], h) + g * h; out.append(torch.argmax(self.head(h), -1).to(torch.int8))
        return torch.stack(out, 1), h


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def int_bits(v: int, n: int) -> np.ndarray:
    """LSB-first bits of a non-negative integer, exactly n of them."""
    assert 0 <= v < (1 << n) if n > 0 else v == 0
    b = np.frombuffer(v.to_bytes((n + 7) // 8, 'little'), np.uint8)
    return np.unpackbits(b, bitorder='little')[:n].copy()


def bits_int(bits: np.ndarray) -> int:
    return int.from_bytes(np.packbits(bits.astype(np.uint8), bitorder='little').tobytes(), 'little')


def true_residues(bits: np.ndarray) -> np.ndarray:
    """per-step residue of the prefix value mod 3 (2^i mod 3 = 1 if i even else 2)."""
    w = np.where(np.arange(bits.size) % 2 == 0, 1, 2).astype(np.int64)
    return (np.cumsum(bits.astype(np.int64) * w) % 3).astype(np.int8)


def onehot(batch_bits: np.ndarray) -> torch.Tensor:  # (B,T) uint8 -> (B,T,2) float32
    x = np.zeros(batch_bits.shape + (2,), np.float32)
    x[..., 0] = batch_bits == 0; x[..., 1] = batch_bits == 1
    return torch.from_numpy(x)


def predict(scripted, seqs: List[np.ndarray], batch: int = 512, chunk: int = 2048) -> List[np.ndarray]:
    """per-step argmax residues for each bit sequence. Sequences are grouped by length (no padding effects at all)."""
    out: List[np.ndarray] = [None] * len(seqs)
    by_len = {}
    for i, s in enumerate(seqs):
        by_len.setdefault(len(s), []).append(i)
    with torch.no_grad():
        for n, idx in sorted(by_len.items()):
            for j in range(0, len(idx), batch):
                ids = idx[j:j + batch]; B = len(ids)
                arr = np.stack([seqs[i] for i in ids]); res = np.empty((B, n), np.int8)
                h = torch.zeros(B, 64)
                for c in range(0, n, chunk):
                    p, h = scripted.predict_chunk(onehot(arr[:, c:c + chunk]), h); res[:, c:c + chunk] = p.numpy()
                for k, i in enumerate(ids):
                    out[i] = res[k]
    return out


def load_checker(seed: int):
    cp = torch.load(HERE / 'checkpoints' / f'checker_s{seed}.pt', weights_only=True)
    m = Checker(); m.load_state_dict(cp['model']); m.eval()
    return torch.jit.script(m), cp


class PMC(ctypes.Structure):
    _fields_ = [('cb', ctypes.wintypes.DWORD), ('PageFaultCount', ctypes.wintypes.DWORD)] + \
               [(n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
                                              'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]


def peak_ram_mib() -> float:
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(PMC), ctypes.wintypes.DWORD]
    assert psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return round(c.PeakWorkingSetSize / 2 ** 20, 1)
