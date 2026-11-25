# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 PMLSTM project author <rickeyernest1990@gmail.com>
# Derived from the PPLSTM research reference: https://github.com/rickey1990/PPLSTM
"""PMLSTM: six-logit column-stochastic transition-monoid reference."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def shift(x):
    return torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)


def apply(A, b):
    return (A @ b.unsqueeze(-1)).squeeze(-1) if A.ndim == b.ndim + 1 else A * b


def state_scan(A, b, c0, serial=False):
    """Inclusive affine scan; serial=True evaluates the same token maps in order."""
    b = torch.cat((b[:, :1] + apply(A[:, :1], c0.expand_as(b[:, :1])), b[:, 1:]), dim=1)
    if serial:
        h = torch.zeros_like(b[:, 0])
        outputs = []
        for t in range(b.shape[1]):
            h = apply(A[:, t], h) + b[:, t]
            outputs.append(h)
        return torch.stack(outputs, dim=1)
    step = 1
    while step < b.shape[1]:
        if A.ndim == b.ndim:
            next_A = A[:, step:] * A[:, :-step]
            next_b = b[:, step:] + A[:, step:] * b[:, :-step]
        else:
            next_A = A[:, step:] @ A[:, :-step]
            next_b = b[:, step:] + (A[:, step:] @ b[:, :-step].unsqueeze(-1)).squeeze(-1)
        A = torch.cat((A[:, :step], next_A), dim=1)
        b = torch.cat((b[:, :step], next_b), dim=1)
        step *= 2
    return b


class PMLSTM(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        if width < 3 or width % 3:
            raise ValueError("width must be a positive multiple of three")
        d = width
        self.d = d
        self.width = d
        self.proj = nn.Linear(d, 4 * d)
        self.mix = nn.Parameter(torch.full((4, d), math.log(3.0)))
        self.c0 = nn.Parameter(torch.linspace(-1.5, 1.5, 3).repeat(d // 3))
        self.ib = nn.Parameter(torch.full((d,), -4.0))
        self.fb = nn.Parameter(torch.full((d,), 4.0))
        self.sign = nn.Linear(d, d)
        self.sel = nn.Linear(d, 2 * d)
        nn.init.zeros_(self.sign.weight)
        nn.init.constant_(self.sign.bias, 2.8)
        nn.init.zeros_(self.sel.weight)
        nn.init.constant_(self.sel.bias, -math.log((math.exp(4) + 1) / 2))

    def forward(self, x, serial=False):
        """x: [batch, time, width]. Returns a tensor of the same shape."""
        if x.ndim != 3 or x.shape[-1] != self.d or x.shape[1] == 0:
            raise ValueError("Expected [batch, nonempty time, width]")
        B, T, D = x.shape
        G = D // 3
        mix = self.mix.sigmoid()
        previous = shift(x)
        u = x * mix[0] + previous * (1 - mix[0])
        gates = self.proj(x).reshape(B, T, 4, D) * mix + self.proj(previous).reshape(B, T, 4, D) * (1 - mix)
        il, fl, zl, ol = gates.unbind(2)
        il = (il + self.ib).reshape(B, T, G, 3).mean(-1)
        fl = (fl + self.fb).reshape(B, T, G, 3).mean(-1)
        z = zl.tanh().reshape(B, T, G, 3)

        # Exponential input gate and sigmoid forget gate, with prefix stabilisation.
        lf = F.logsigmoid(fl)
        prefix = lf.cumsum(1)
        st = prefix + torch.cummax(torch.cat((torch.zeros_like(prefix[:, :1]), il - prefix), 1), 1).values[:, 1:]
        f = (lf + shift(st) - st).exp()
        i = (il - st).exp()
        s = self.sign(u).tanh().reshape(B, T, G, 3)

        # Each source column uses [0, a, b] logits in relative destination order.
        logits = self.sel(u).reshape(B, T, G, 3, 2)
        weights = torch.cat((torch.zeros_like(logits[..., :1]), logits), -1).softmax(-1)
        M = torch.stack([torch.stack([weights[..., col, (row-col) % 3]
                                      for col in range(3)], -1)
                         for row in range(3)], -2)

        # A_t = f_t M_t diag(s_t); b_t = i_t z_t.
        A = f[..., None, None] * M * s[..., None, :]
        c = state_scan(A, i[..., None] * z, self.c0.tanh().reshape(1, 1, G, 3), serial)
        n = state_scan(f, i, torch.ones(1, 1, G, device=x.device, dtype=x.dtype), serial)
        return ol.sigmoid() * (c / n[..., None].clamp_min(1e-12)).reshape(B, T, D)

    def serial_reference(self, x):
        """Serial state update with identical precomputed token maps."""
        return self.forward(x, serial=True)


PM = PMLSTM

if __name__ == "__main__":
    cell = PMLSTM(36)
    print(cell(torch.randn(2, 17, 36)).shape)
