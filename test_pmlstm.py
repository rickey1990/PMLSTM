# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 PMLSTM project author <rickeyernest1990@gmail.com>
"""CPU checks for both PM implementations; no dataset or GPU required."""
import copy
import torch
from pmlstm import PMLSTM
from pmlstm_27 import PMLSTM27


def check(cls):
    torch.manual_seed(13)
    parallel = cls(36).double()
    serial = copy.deepcopy(parallel)
    x = torch.randn(2, 65, 36, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    a, b = parallel(x), serial.serial_reference(y)
    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-10, rtol=1e-9)
    errors = []
    for p, q in zip(parallel.parameters(), serial.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-9)
        errors.append((p.grad - q.grad).abs().max().item())
    with torch.no_grad():
        altered = x.detach().clone()
        altered[:, 32:] += torch.randn_like(altered[:, 32:])
        torch.testing.assert_close(parallel(x)[:, :32], parallel(altered)[:, :32], atol=1e-10, rtol=1e-9)
        assert torch.isfinite(parallel(torch.randn(1, 8192, 36, dtype=torch.float64))).all()
    print(f'{cls.__name__}: PASS | params={sum(p.numel() for p in parallel.parameters())} | output error={(a-b).abs().max().item():.3g} | max parameter gradient error={max(errors):.3g}')

if __name__ == '__main__':
    torch.set_num_threads(1)
    check(PMLSTM)
    check(PMLSTM27)
