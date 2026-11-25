# PMLSTM: a second parallel LSTM cell using transition monoids instead of permutations

I've been experimenting with a second alternative to sLSTM, following on from my earlier PPLSTM (Parallel Permutation LSTM) project. This one is called **PMLSTM (Parallel Monoid Long Short-Term Memory)**.

PPLSTM uses a learned mixture over six permutations of three memory channels. PMLSTM started with the full 27-element transition monoid, which includes those six permutations and another 21 non-bijective transitions. It turns out that the same family of transition matrices can be represented directly using just six learned logits per group, so the final PM has the same parameter count as PP.

The rest of the recurrence stays the same: signed carry, input/forget gates, stabilisation, normalisation and an associative prefix scan. It is a new recurrent cell, rather than an exact parallel implementation of sLSTM.

In a small CPU experiment, each model saw roughly five million training tokens. Standalone PP and PM achieved held-out perplexity of 71.96 and 72.60 at the same parameter count. With mLSTM, the scores were 57.39 and 58.05, respectively. I'm not claiming a win for PM; what interests me is that two different structured transition mechanisms achieved very similar results.

Both PM implementations are available as standalone PyTorch files, along with a short proof-of-concept report and numerical checks. This is still early research: the main language comparison is one seed, CPU speed varies by configuration, PM GPU benchmarks are outstanding, and direct memory gradient decay remains a concern. The larger transition family is not automatically better at every state-tracking task.

I'd particularly welcome feedback on the transition-monoid formulation, the six-logit parameterisation, and what tests would best distinguish PP and PM.

Code: AGPL-3.0-only; separate commercial licensing enquiries welcome. The mLSTM architecture is credited to NX-AI/xLSTM (Apache-2.0 upstream code).

PM repository: https://github.com/rickey1990/PMLSTM

Original PPLSTM project: https://github.com/rickey1990/PPLSTM
