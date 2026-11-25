# PMLSTM — Parallel Monoid Long Short-Term Memory

**Research prototype · November 2026**

PMLSTM (PM) is a parallel-scannable recurrent memory cell developed from [PPLSTM (PP)](https://github.com/rickey1990/PPLSTM). It is being investigated as another practical alternative to the sequential sLSTM component of xLSTM-style models. It is not mathematically equivalent to sLSTM.

## Two implementations

| File | Description |
|---|---|
| [`pmlstm.py`](pmlstm.py) | **PM — the main implementation.** Six learned logits per three-channel group; three column-wise softmax operations. |
| [`pmlstm_27.py`](pmlstm_27.py) | Original formulation: a learned mixture over all 27 deterministic three-channel transitions. Retained for comparison and research history. |

The original formulation is $M_t=\sum_{j=1}^{27}w_{t,j}T_j$. All six permutations are included in the 27-map transition monoid. Its convex hull is the set of column-stochastic 3 × 3 matrices. The final formulation uses two learned logits per source column, with a fixed zero logit for the identity route:

$
M_t[r,j]=\mathrm{softmax}(0,a_{t,j},b_{t,j})[(r-j)\bmod 3].
$

This preserves the attainable transition-matrix family at each token (with boundary matrices approached in the softmax limit), not necessarily the same learned function or optimisation dynamics. The signed memory recurrence remains:

$
c_t=f_tM_t\mathrm{diag}(s_t)c_{t-1}+i_tz_t,
$

with the PPLSTM stabiliser, normaliser and associative affine prefix scan. The final PM has the same selector parameter count as PP. See the [proof of concept](PMLSTM_Proof_of_Concept.pdf) for the full equations.

## Quick start

```bash
pip install -r requirements.txt
python test_pmlstm.py
```

```python
import torch
from pmlstm import PMLSTM

cell = PMLSTM(width=36)
x = torch.randn(2, 128, 36)
y = cell(x)
print(y.shape)  # torch.Size([2, 128, 36])
```

For the historical cell, use `from pmlstm_27 import PMLSTM27`. Both modules are self-contained and use plain PyTorch; width must be divisible by three. Both expose `serial_reference(x)` for numerical comparison with the parallel state scan. No custom CUDA kernel or streaming state API is included. The final PM constructs its column-stochastic matrix directly, without a permutation bank or enumeration of the 27 maps. For older standalone PM checkpoints only, remove the unused `bank` entry from the state dictionary before loading (`state_dict.pop('bank', None)` followed by `cell.load_state_dict(state_dict)`). The same applies to the corresponding PM module's prefixed buffer key in a full-model checkpoint. All trainable parameters and initial values are unchanged.

## Early results — final six-logit PM

Single-seed CPU study, 5,001,216 training-token presentations per model, context 32, width 36. The replacement Wikipedia-derived corpus is not the original PP proof-of-concept corpus.

| Model | Parameters | Validation PPL ↓ | Test PPL ↓ |
|---|---:|---:|---:|
| PP | 34,848 | 74.22 | 71.96 |
| PM | 34,848 | 72.91 | 72.60 |
| mLSTM + PP | 47,248 | 59.57 | 57.39 |
| mLSTM + PM | 47,248 | 60.17 | 58.05 |
| Transformer | 30,600 | 97.78 | 98.67 |

PM and PP had comparable held-out language quality at identical parameter counts, both alone and with mLSTM. CPU speed varied across configurations; GPU throughput and peak GPU memory have not yet been measured for PM. The Transformer control is smaller and lacks explicit positional embeddings; this is not a claim of superiority over Transformers. Results are preliminary, not statistical equivalence or a large-scale benchmark.

The included CPU tests verify serial/parallel agreement, input and parameter gradients, causality, and finite outputs at 8,192 tokens. The column-stochastic carry bounds the isolated recurrent memory product in L1 norm, but does not rule out vanishing gradients or instability elsewhere in the network. Trained-checkpoint diagnostics found substantial direct-memory gradient decay in both PP and PM. Long-context perplexity is not proof of long-distance recall.

- [Proof of concept (PDF)](PMLSTM_Proof_of_Concept.pdf)
- [Language results](results/language.json), [CPU speed](results/speed_cpu.json), [initial checks](results/initial_checks.json), [configuration](results/config.json), [gradient diagnostics](results/gradient_audit.json)

This is a cell release, not the full training reproduction archive: dataset, tokeniser, benchmark scripts and trained checkpoints are not included. The published results were obtained with the original experimental modules; the two standalone modules retain their equations and trainable parameter layout. The cleaned six-logit module was checked against the previous release in float32 and float64 on CPU, with exactly equal outputs, input gradients and parameter gradients for the tested inputs and weights.

## Licence and attribution

PMLSTM and PPLSTM-derived source: **GNU AGPL-3.0-only**. Commercial use under AGPL is permitted without payment. See [LICENSE](LICENSE) and [commercial licensing](COMMERCIAL_LICENSING.md) for separate licensing enquiries.

mLSTM originates from [NX-AI/xlstm](https://github.com/NX-AI/xlstm), licensed under [Apache-2.0](https://github.com/NX-AI/xlstm/blob/main/LICENSE). The reported experiments used a reference mLSTM implementation, not NX-AI's optimised production kernels. Neither standalone PM file includes NX-AI source code. See [third-party attribution](THIRD_PARTY_NOTICES.md).

Related work: [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517) (Beck et al., 2024).

Contact: **rickeyernest1990@gmail.com**.
