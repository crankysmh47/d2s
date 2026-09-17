# Parameter-Preserving Dense → Sparse MoE

> Historical roadmap, archived 2026-09-08. See ../../plan.md for the active research direction. This document records the original design; its unexecuted phases are not an active queue. Phase 1A and 1A.5 results remain authoritative in PROJECT_LOG.md and their immutable artifacts.

## Revised Research and Engineering Plan

**Working names:** PP-MoE / Contribution-Gated MoE / Function-Preserving MoEfication
**First real experimental model:** Qwen3.5-0.8B-Base
**First hardware target:** Single GPU, ≤16 GB VRAM
**Controlled custom SLM:** Mini-Qwen3.8, after initial oracle validation
**Eventual target:** Qwen3.8-27B text FFNs
**Primary objective:** Preserve approximately the same total model parameters while reducing active FFN computation per token with minimal quality loss.

---

# 0. Major revisions from the previous plan

The revised project makes six important changes.

## Revision 1 — We do not train our own SLM first

The first meaningful experiment uses the already-pretrained:

**Qwen3.5-0.8B-Base**

Its text model has:

- 24 layers;
- hidden size 1024;
- FFN intermediate size 3584;
- 6 repetitions of:
  - 3 linear-attention / Gated DeltaNet layers;
  - 1 full-attention layer;
- 8 query attention heads;
- 2 KV heads;
- RMSNorm;
- SiLU FFNs;
- tied token embeddings.

The official model card reports a 0.8B language model and the official checkpoint is approximately 1.77 GB. citeturn571909search1turn480633view1turn571909search5

This lets us test the central hypothesis without first spending billions of tokens pretraining a dense model.

---

## Revision 2 — The first experiment contains no learned router

The first meaningful experiment is:

```text
pretrained dense SLM
        ↓
exact FFN partition
        ↓
all-expert equivalence
        ↓
oracle routing
        ↓
quality / active-compute curve
```

We do not train a router until the oracle demonstrates that worthwhile conditional sparsity actually exists.

---

## Revision 3 — We explicitly distinguish two kinds of oracle

A local teacher-forced oracle can produce overly optimistic results because every layer receives the original dense hidden state.

We therefore distinguish:

```text
Local Oracle
    ↓
"Is this individual dense FFN intrinsically sparsifiable?"

Rollout Oracle
    ↓
"Does sparsification still work after errors propagate?"
```

Both are required.

---

## Revision 4 — Expert cancellation becomes measurable

Instead of merely noting that two expert outputs could cancel, the oracle formulation will explicitly account for:

\[
E_i^\top E_j
\]

pairwise interaction terms.

We will also log expert interaction and oracle co-selection matrices.

---

## Revision 5 — Contribution-vector statistics become streaming and projected

We do not materialize gigantic neuron × token × hidden-size tensors.

We exploit:

\[
c_j(x)=W_{d,:,j}h_j(x)
\]

and pre-project the static down-projection vector once.

This makes functional clustering practical.

---

## Revision 6 — Variable-K is removed from V1

Version 1 is strictly:

\[
\boxed{\text{fixed K}}
\]

Variable-K routing becomes a later research phase only after:

- partitioning works;
- oracle routing works;
- a fixed-K learned router works.

---

# 1. Research objective

We are not building the standard capacity-expanded MoE.

Conventional sparse upcycling often looks approximately like:

```text
Dense model
     ↓
duplicate / expand FFNs
     ↓
much larger total parameter count
     ↓
activate a small number of experts/token
```

Our approach is:

```text
Dense model
     ↓
reorganize existing FFN neurons
     ↓
same FFN parameter budget
     ↓
conditionally execute neuron groups
```

The target is therefore:

\[
P_{\text{PP-MoE,total}}
\approx
P_{\text{Dense,total}}
\]

while:

\[
P_{\text{PP-MoE,active}}
<
P_{\text{Dense,active}}
\]

and:

\[
Q_{\text{PP-MoE}}
\approx
Q_{\text{Dense}}
\]

The ideal optimization problem is:

\[
\boxed{
\min \mathbb E[C(x,l)]
}
\]

subject to:

\[
\boxed{
D(M_{\text{sparse}},M_{\text{dense}})
\le \epsilon
}
\]

where \(C(x,l)\) is the conditional FFN compute spent on token \(x\) at layer \(l\).

---

# 2. Why this is mathematically possible

Consider a SwiGLU FFN:

\[
F(x)
=
W_d
\left[
\operatorname{SiLU}(W_gx)
\odot
(W_ux)
\right]
\]

with:

\[
W_g,W_u
\in
\mathbb R^{d_{\mathrm{ff}}\times d_{\mathrm{model}}}
\]

and:

\[
W_d
\in
\mathbb R^{d_{\mathrm{model}}\times d_{\mathrm{ff}}}.
\]

For intermediate neuron \(j\):

\[
h_j(x)
=
\operatorname{SiLU}(W_{g,j:}x)
(W_{u,j:}x)
\]

and its contribution to the FFN output is:

\[
c_j(x)
=
W_{d,:,j}h_j(x).
\]

Therefore:

\[
\boxed{
F(x)=\sum_{j=1}^{d_{\mathrm{ff}}}c_j(x)
}
\]

This is an algebraic identity.

The project is built on that identity.

Earlier MoEfication work similarly showed that pretrained Transformer FFNs can be partitioned into experts without increasing parameter count, reporting substantial conditional FFN sparsity on its evaluated models. citeturn571909academia24

---

# 3. The atomic SwiGLU neuron

A SwiGLU intermediate neuron is represented by three inseparable pieces:

```text
gate_proj row j
up_proj   row j
down_proj column j
```

These three tensors collectively define neuron \(j\).

They must always:

- be clustered together;
- be permuted together;
- be assigned to the same expert.

We never independently cluster `gate_proj`, `up_proj`, and `down_proj`.

For expert index set \(I_e\):

\[
E_e(x)
=
W_{d,:,I_e}
\left[
\operatorname{SiLU}(W_{g,I_e,:}x)
\odot
W_{u,I_e,:}x
\right].
\]

If:

\[
I_1,\dots,I_N
\]

form a disjoint partition of all intermediate neurons, then:

\[
\boxed{
F(x)=\sum_{e=1}^{N}E_e(x)
}
\]

exactly.

---

# 4. Bias handling

Qwen-style FFNs make this especially clean, but the general implementation should support biases.

If gate/up projections have per-neuron biases, those biases move with their neurons.

If the down projection has one output bias \(b_d\), then:

\[
F(x)
=
b_d
+
\sum_eE_e(x).
\]

The down-projection bias must therefore be added **once**, not duplicated into every expert.

---

# 5. Function-preserving conversion

Before introducing routing, the converted model must have a special mode:

```text
all_experts = True
```

Then:

\[
F_{\text{converted}}(x)
=
F_{\text{dense}}(x)
\]

up to ordinary floating-point accumulation differences.

This is our first hard invariant.

---

# 6. Verification hierarchy

We verify conversion at progressively larger scopes.

## Test A — single neuron permutation

Permute the FFN intermediate dimension.

Apply the identical permutation to:

```text
gate_proj rows
up_proj rows
down_proj columns
```

Dense output must remain unchanged.

---

## Test B — one FFN split

Partition one FFN.

Compute:

\[
\sum_eE_e(x)
\]

and compare to the original dense FFN.

---

## Test C — one Transformer layer

Replace one FFN with the expert representation.

All experts active.

Compare complete layer output.

---

## Test D — entire language model

Convert every FFN.

All experts active.

Compare:

- every pre-MLP normalized state;
- every FFN output;
- every layer output;
- final hidden state;
- logits;
- perplexity.

---

# 7. RMSNorm and numerical precision verification

The router consumes the same normalized hidden state that enters the FFN:

\[
x_l^{\text{norm}}
=
\operatorname{RMSNorm}(x_l).
\]

The FFN neuron permutation occurs in:

\[
d_{\mathrm{ff}}
\]

space.

It does not alter the residual dimension:

\[
d_{\mathrm{model}}.
\]

Therefore RMSNorm itself should be unaffected.

The flow is:

```text
residual stream
      ↓
RMSNorm
      ↓
normalized hidden
   ┌──┴─────────────┐
   ↓                ↓
router           experts
```

Token dispatch simply copies/gathers this hidden vector; the gather operation itself does not introduce arithmetic rounding.

The meaningful BF16 differences arise from:

- splitting one large GEMM into multiple GEMMs;
- changed accumulation order;
- expert-output summation;
- fused vs unfused kernels.

---

# 8. Numerical tests

In FP32 reference mode:

\[
\text{relative error}
\lesssim
10^{-5}
\]

should be achievable.

In BF16, log:

- max absolute error;
- RMS error;
- normalized MSE;
- cosine similarity;
- final logit KL.

Also compare:

```text
BF16 expert accumulation
vs
FP32 expert accumulation → BF16 cast
```

If FP32 expert summation materially improves model-level drift, the eventual sparse kernel may accumulate expert outputs in FP32.

---

# 9. Development phases

The revised project has five major phases.

```text
Phase 0
Tiny synthetic model
      ↓
prove conversion implementation

Phase 1
Qwen3.5-0.8B-Base
      ↓
oracle-only SLM study

Phase 2
Qwen3.5-0.8B learned routing
      ↓
fixed-K router + distillation

Phase 3
Custom Mini-Qwen3.8
      ↓
controlled from-scratch science

Phase 4
Qwen3.8-27B
      ↓
large-scale target
```

---

# 10. Phase 0 — tiny synthetic validation

Use a tiny SwiGLU Transformer.

For example:

```yaml
hidden_size: 128
intermediate_size: 512
layers: 2
heads: 4
vocab: 4096
```

It does not need to be a good language model.

Purpose:

- verify neuron extraction;
- verify permutation;
- verify checkpoint conversion;
- verify expert packing;
- verify exact all-expert mode;
- verify gradients;
- implement oracle subset enumeration;
- test BF16 vs FP32 behavior.

This phase should be extremely short.

No research conclusions are drawn from it.

---

# 11. Phase 1 — first real SLM

The first real research model is:

\[
\boxed{\text{Qwen3.5-0.8B-Base}}
\]

It is already pretrained, small enough for our hardware budget, and architecturally close to the eventual Qwen3.8 target. Its official text configuration uses hidden size 1024, FFN width 3584, 24 layers, and a repeated three-linear-attention/one-full-attention pattern. citeturn571909search1turn480633view1

The model contains a vision stack as well; Phase 1 uses **text inputs only** and leaves the vision components untouched.

---

# 12. Phase 1A — simplest possible partition

Do not begin with:

- shared experts;
- contribution clustering;
- sophisticated gates.

Start with the fewest assumptions possible.

Qwen3.5-0.8B has:

\[
d_{\mathrm{ff}}=3584.
\]

Divide it evenly into eight experts:

\[
3584/8=448.
\]

Thus:

```text
E0 = 448 neurons
E1 = 448
E2 = 448
E3 = 448
E4 = 448
E5 = 448
E6 = 448
E7 = 448
```

No shared expert.

Use a random balanced partition.

This is our baseline.

---

# 13. Phase 1A active fractions

With no shared expert:

\[
r(K)=K/8.
\]

Therefore:

| K | Active FFN |
|---|---:|
| 8 | 100% |
| 7 | 87.5% |
| 6 | 75% |
| 5 | 62.5% |
| 4 | 50% |
| 3 | 37.5% |
| 2 | 25% |
| 1 | 12.5% |

This gives a very clean first oracle curve.

---

# 14. The first real question

Before any learned router:

> With a random partition and an omniscient router, how much dense FFN computation is actually redundant on each token?

This gives us a lower-bound decomposition.

If even random groups can reach useful sparsity, that is strong evidence.

If random groups fail, smarter grouping may still work.

---

# 15. Local teacher-forced oracle

For dense pre-MLP input:

\[
x_l^T
\]

compute:

\[
y^T=F_l(x_l^T).
\]

Compute every expert output on the same teacher state:

\[
e_i=E_i(x_l^T).
\]

For binary expert mask:

\[
m\in\{0,1\}^{N}
\]

with:

\[
\sum_i m_i=K,
\]

define:

\[
\hat y(m)=\sum_i m_ie_i.
\]

Then:

\[
m^*
=
\arg\min_m
\|y^T-\hat y(m)\|^2.
\]

This asks:

> On the exact dense hidden state, what is the best K-expert reconstruction?

It measures the intrinsic sparsifiability of one FFN.

---

# 16. Exact subset oracle for N = 8

With eight experts:

\[
2^8=256
\]

total subsets.

For fixed K:

\[
\binom8K
\]

is tiny.

For example:

\[
\binom82=28
\]

and:

\[
\binom84=70.
\]

Therefore the initial oracle is **exact**, not greedy.

This is a major advantage of beginning with eight experts.

---

# 17. Efficient exact oracle scoring

We do not need to repeatedly construct every candidate output.

Let:

\[
r=y^T-y_{\text{shared}}
\]

where initially:

\[
y_{\text{shared}}=0.
\]

Let expert-output matrix conceptually be:

\[
E=[e_1,\dots,e_N].
\]

Then:

\[
L(m)
=
\left\|
r-\sum_i m_ie_i
\right\|^2.
\]

Expand:

\[
L(m)
=
\|r\|^2
-
2b^\top m
+
m^\top Gm
\]

where:

\[
b_i=r^\top e_i
\]

and:

\[
G_{ij}=e_i^\top e_j.
\]

For \(N=8\):

- compute \(b\);
- compute the \(8\times8\) Gram matrix \(G\);
- evaluate all candidate masks extremely cheaply.

This exact expression automatically accounts for:

- expert magnitude;
- redundancy;
- synergy;
- cancellation.

---

# 18. Expert cancellation is therefore explicit

Suppose:

\[
e_1\approx-e_2.
\]

Then:

\[
e_1^\top e_2<0.
\]

That negative interaction appears directly in:

\[
m^\top Gm.
\]

Thus oracle routing does not incorrectly choose experts purely because they have large individual norms.

---

# 19. Interaction diagnostics

For every layer, collect mean cosine interaction:

\[
C_{ij}
=
\mathbb E_x
\left[
\frac{
e_i^\top e_j
}{
\|e_i\|\|e_j\|+\epsilon
}
\right].
\]

Produce an 8×8 matrix.

Interpretation:

\[
C_{ij}\approx1
\]

strongly redundant/aligned.

\[
C_{ij}\approx0
\]

weak interaction.

\[
C_{ij}<0
\]

partial cancellation.

This becomes a first-class diagnostic.

---

# 20. Oracle co-selection matrix

Also record:

\[
S_{ij}
=
P(i\in S^*,j\in S^*).
\]

This tells us which experts the oracle frequently needs together.

If:

```text
E2 and E5
```

are almost always selected together, that may mean the partition is too fragmented.

Oracle co-selection can later become a signal for re-clustering.

---

# 21. The second oracle: rollout oracle

A local oracle receives teacher hidden states at every layer.

That can hide error accumulation.

In deployment, once layer 3 makes an approximation error:

\[
x_4^S\neq x_4^T.
\]

Therefore the layer-4 sparse FFN receives a different input.

We require a second oracle.

---

# 22. Greedy rollout oracle

Run the sparse model sequentially.

At layer \(l\), current student state is:

\[
x_l^S.
\]

The non-FFN portion of the layer operates normally.

For the FFN, evaluate all fixed-K subsets and select the subset minimizing distance to the **teacher next-layer state**:

\[
S_l^*
=
\arg\min_S
\|
x_{l+1}^S(S)-x_{l+1}^T
\|^2.
\]

Then continue using that selected sparse state.

This answers:

> Can a layer-local optimal routing policy keep the whole trajectory near the teacher once previous errors exist?

This is more realistic than the teacher-forced oracle.

---

# 23. Three oracle levels

The final terminology should be:

### Oracle A — local FFN oracle

Teacher input, compare FFN output.

Purpose:

```text
intrinsic FFN sparsifiability
```

### Oracle B — one-step rollout oracle

Student current state, choose route minimizing next hidden-state divergence.

Purpose:

```text
error propagation robustness
```

### Oracle C — global oracle

Optimize routing for final logits / sequence loss over all layers.

This is combinatorial and not required initially.

It can later be approximated with:

- beam search;
- differentiable routing;
- dynamic programming approximations;
- reinforcement-style search.

V1 requires A and B only.

---

# 24. Phase 1 baseline deliverables

Before any routing training, produce:

```text
Plot 1
K vs local FFN reconstruction error

Plot 2
active FFN fraction vs rollout hidden-state error

Plot 3
active FFN fraction vs perplexity

Plot 4
layer × K sparsifiability heatmap

Plot 5
expert interaction/cancellation heatmaps
```

These plots constitute the first genuine project result.

---

# 25. First go/no-go gate

Suppose K=4:

\[
50\%
\]

active FFN.

We would like:

- very low local FFN error;
- limited rollout error;
- near-dense perplexity.

If even the **oracle** causes severe quality degradation at 50%, then there is no reason to train a router yet.

Instead improve the decomposition.

---

# 26. Phase 1B — intelligent expert formation

After the random baseline, introduce intelligent grouping.

Keep:

- N = 8;
- fixed K;
- no learned router.

Only change how neurons are grouped.

Compare:

1. random balanced;
2. weight clustering;
3. activation clustering;
4. co-activation clustering;
5. contribution-magnitude clustering;
6. projected contribution clustering.

The winner is determined by oracle curves.

---

# 27. Neuron activation

For neuron \(j\):

\[
h_j(x)
=
\operatorname{SiLU}(W_{g,j:}x)
(W_{u,j:}x).
\]

We can log:

- mean \(|h_j|\);
- RMS activation;
- activation frequency over threshold;
- quantiles;
- top-token frequency.

---

# 28. Neuron contribution

More important is:

\[
c_j(x)
=
W_{d,:,j}h_j(x).
\]

A scalar contribution magnitude can be:

\[
I_j(x)
=
\|c_j(x)\|_2.
\]

Because:

\[
c_j(x)
=
W_{d,:,j}h_j(x),
\]

we have:

\[
I_j(x)
=
|h_j(x)|
\|W_{d,:,j}\|_2.
\]

Therefore contribution magnitude can be computed very cheaply.

This is a useful simplification.

---

# 29. Cheap contribution-vector sketches

The directional contribution is:

\[
c_j(x)
=
W_{d,:,j}h_j(x).
\]

Choose a fixed random projection:

\[
R\in\mathbb R^{r\times d_{\mathrm{model}}}
\]

with small:

\[
r\in\{32,64,128\}.
\]

Precompute once per neuron:

\[
p_j=RW_{d,:,j}.
\]

Then:

\[
Rc_j(x)
=
p_jh_j(x).
\]

Thus we never construct a full \(d_{\mathrm{model}}\)-dimensional contribution vector for every token/neuron pair.

We only multiply:

```text
static projected vector
×
scalar activation
```

This makes contribution-vector statistics practical.

---

# 30. Streaming statistics

Do not store:

\[
[\text{number of tokens}]
\times
[d_{\mathrm{ff}}]
\times
[d_{\mathrm{model}}].
\]

Maintain streaming sketches.

For every neuron:

```text
activation count
mean |h|
activation variance
mean contribution norm
variance of contribution
selected quantiles
compact projected signature
top-activation count
```

For co-activation:

- retain only top-M important neurons/token;
- update a sparse co-occurrence graph.

---

# 31. Calibration data schedule

Do not immediately process tens of millions of tokens.

Use progressive calibration.

```text
100K tokens
    ↓
first statistics + oracle

500K
    ↓
stability comparison

1M
    ↓
recluster and compare oracle

5M
    ↓
only if metrics still change materially

10M+
    ↓
only with demonstrated benefit
```

The amount of calibration data is itself an ablation.

---

# 32. Weight clustering

Neuron descriptor may include compressed versions of:

\[
W_{g,j:}
\]

\[
W_{u,j:}
\]

\[
W_{d,:,j}.
\]

Normalize each component before concatenation.

Possible preprocessing:

- random projection;
- PCA;
- per-block normalization.

Use balanced clustering so all expert widths remain identical or nearly identical.

---

# 33. Activation clustering

For each neuron, build a compact signature indicating:

> On which kinds of tokens is this neuron active?

Possible representation:

\[
a_j
=
[\tilde h_j(x_1),\dots,\tilde h_j(x_M)]
\]

on a sampled token set.

Or use streaming random sketches rather than storing the full vector.

---

# 34. Functional / contribution clustering

Our primary advanced hypothesis is:

> Neurons should share an expert when the dense teacher tends to need their output contributions together.

This is stronger than:

> Neurons should share an expert because their weights look similar.

The oracle curve determines whether this hypothesis is correct.

---

# 35. Iterative oracle-guided clustering

A later clustering procedure can use the oracle itself.

```text
initial partition
      ↓
oracle routing
      ↓
co-selection statistics
      ↓
identify groups consistently needed together
      ↓
repartition neurons
      ↓
run oracle again
```

This can be repeated until oracle quality stops improving.

This is a promising research direction but is **not required for V1**.

---

# 36. Phase 1C — shared expert

Only after establishing the no-shared baseline do we introduce an always-active shared subset.

For Qwen3.5-0.8B:

\[
d_{\mathrm{ff}}=3584.
\]

A particularly clean hardware-aligned split is:

\[
3584
=
512+8\times384.
\]

So:

```text
shared = 512

E0 = 384
E1 = 384
E2 = 384
E3 = 384
E4 = 384
E5 = 384
E6 = 384
E7 = 384
```

---

# 37. Active ratios with the shared expert

K=1:

\[
\frac{512+384}{3584}=25\%.
\]

K=2:

\[
\frac{512+768}{3584}
\approx35.7\%.
\]

K=3:

\[
\frac{512+1152}{3584}
\approx46.4\%.
\]

K=4:

\[
\frac{512+1536}{3584}
\approx57.1\%.
\]

This gives an attractive experimental range.

---

# 38. Selecting shared neurons

Potential shared-neuron scores include:

## Activation frequency

\[
f_j
=
P(|h_j|>\tau_h).
\]

## Mean activation

\[
A_j
=
\mathbb E[|h_j|].
\]

## Mean contribution

\[
C_j
=
\mathbb E[\|c_j(x)\|].
\]

## Contribution frequency

\[
q_j
=
P(\|c_j(x)\|>\tau_c).
\]

## Hybrid importance

For example:

\[
S_j
=
q_j^\alpha C_j^\beta.
\]

Test these experimentally.

---

# 39. Why shared experts matter

If one neuron is useful on 95% of tokens and belongs to routed expert 3:

```text
almost every token must select expert 3
```

This effectively wastes one routing slot.

Moving universal computation into a shared block leaves routed experts more conditional.

This idea is also consistent with later dense-to-MoE restructuring work that separates broadly active and conditional FFN neurons.

---

# 40. Phase 1 final decomposition selection

At the end of the oracle-only phase we should have selected:

```text
shared width
number of experts
neuron partition
per-layer K candidate
```

using **oracle results only**.

No learned router should influence architectural selection.

This avoids conflating:

```text
bad decomposition
```

with:

```text
bad router
```

---

# 41. Layerwise sparsifiability

For layer \(l\):

\[
E_l(K)
=
\mathbb E_x
\left[
\frac{
\|F_l^T-F_{l,K}^{oracle}\|
}{
\|F_l^T\|+\epsilon
}
\right].
\]

We expect some layers to be harder.

Possible result:

```text
layer  0 → top-2 viable
layer  1 → top-2 viable
layer  2 → top-3
layer  3 → top-4
layer  4 → top-2
...
```

V1 benchmarking can still use the same K everywhere for clean comparison.

Later deployment may use layer-specific K.

---

# 42. Phase 2 — learned routing

Only after the oracle passes our viability threshold do we train routing.

The primary router question is:

> Can the current hidden state predict the oracle-selected subset cheaply?

---

# 43. Why standard MoE softmax output weighting is not natural

Conventional MoE:

\[
y
=
\sum_i p_iE_i(x)
\]

with:

\[
\sum_i p_i=1
\]

is appropriate when each expert is an alternative full transformation.

Our experts are pieces of one additive function.

The original dense computation is:

\[
F(x)=\sum_iE_i(x).
\]

If every expert is needed but softmax produces:

\[
p_i=1/N,
\]

then:

\[
\sum_i p_iE_i(x)
=
\frac1N F(x).
\]

That is incorrect.

Therefore **router probability and expert-output scaling must be conceptually separated**.

---

# 44. Default deployment semantics

The default sparse FFN is:

\[
F_S(x)
=
E_{\mathrm{shared}}(x)
+
\sum_{i\in S(x)}E_i(x).
\]

Selected experts receive multiplier:

\[
1.
\]

The router chooses the set.

It does not normalized-weight the expert outputs.

---

# 45. Gated-attention principle

The useful lesson from gated attention is:

> Learn a continuous relevance/control signal before committing to a discrete sparse computation.

For expert \(i\):

\[
s_i=r_i(x)
\]

and:

\[
g_i=\sigma(s_i).
\]

Interpretation:

> How necessary is expert \(i\) for this token?

This is different from:

> Which expert wins?

---

# 46. Exactness-preserving gate training

An improvement over the previous plan:

We do **not** need to multiply all expert outputs by 0.99 gates at the start.

That would technically break exact equivalence.

Instead separate:

```text
router score
```

from:

```text
execution mask
```

During initial router training:

```text
all experts execute with weight 1
```

while the router independently learns oracle labels.

Thus the model remains exactly dense-equivalent.

Only later does the router control execution.

---

# 47. Optional anchored soft gate

If soft output gating is useful during sparsification, define:

\[
\tilde g_i
=
1-\lambda(t)(1-\sigma(s_i)).
\]

When:

\[
\lambda(0)=0,
\]

then:

\[
\tilde g_i=1
\]

exactly for every expert.

As training progresses:

\[
\lambda(t)\rightarrow1,
\]

and:

\[
\tilde g_i\rightarrow\sigma(s_i).
\]

This gives an exact dense starting point without needing extreme sigmoid biases.

---

# 48. Router R1 — linear top-K

\[
s=W_rx+b.
\]

Select:

\[
S=\operatorname{TopK}(s,K).
\]

The expert outputs are summed with unit weight.

This is the simplest learned baseline.

---

# 49. Router R2 — independent sigmoid

\[
g=\sigma(W_rx+b).
\]

For fixed K:

\[
S=\operatorname{TopK}(g,K).
\]

The sigmoid values provide interpretable independent necessity scores.

---

# 50. Router R3 — query/expert-key gated router

Inspired by gated-attention principles:

\[
q=W_qx.
\]

Each expert has descriptor:

\[
k_i.
\]

Then:

\[
s_i
=
\frac{q^\top k_i}{\sqrt{d_r}}
+b_i
\]

and:

\[
g_i=\sigma(s_i).
\]

This asks whether the current token state matches the functional characteristics of each expert.

---

# 51. Router R4 — small MLP

If the relationship is not linearly separable:

\[
s=W_2\phi(W_1x).
\]

Keep the bottleneck small.

The router parameter count should remain negligible relative to the model.

---

# 52. Router R5 — subset classifier

This is an important new diagnostic for cancellation/interactions.

For fixed:

\[
N=8,\quad K=2,
\]

there are only:

\[
\binom82=28
\]

possible expert sets.

We can directly train a classifier over 28 subsets.

For K=3:

\[
\binom83=56.
\]

This explicitly models set-level interactions.

It is not necessarily the scalable final router, but it gives a useful learned upper baseline.

If:

```text
subset classifier ≫ independent sigmoid
```

then expert interactions are a major routing issue.

---

# 53. Teacher-derived router labels

The exact oracle provides labels for free.

For fixed K:

\[
t_i
=
\begin{cases}
1 & i\in S^*\\
0 & \text{otherwise}.
\end{cases}
\]

For sigmoid router:

\[
L_{\text{route-BCE}}
=
-\sum_i
[
t_i\log g_i
+
(1-t_i)\log(1-g_i)
].
\]

---

# 54. Reconstruction loss handles interactions

Do not train the router solely on independent BCE labels.

Also directly optimize the chosen expert combination.

For soft/hard mask \(m_i\):

\[
L_{\text{recon}}
=
\left\|
F^T(x)
-
E_s(x)
-
\sum_i m_iE_i(x)
\right\|^2.
\]

This automatically captures cancellation and pair interactions.

Therefore:

\[
L_{\text{router}}
=
\lambda_RL_{\text{route-BCE}}
+
\lambda_FL_{\text{recon}}.
\]

This is the recommended V1 router objective.

---

# 55. Ranking supervision

The oracle can also assign an expert marginal value.

Given current selected set \(S\):

\[
\Delta_i
=
L(S)-L(S\cup\{i\}).
\]

Use these values for:

- pairwise ranking;
- margin ranking;
- calibration diagnostics.

Not required for first router run.

---

# 56. Router input

Use the exact state that feeds the dense FFN:

\[
x^{router}
=
x^{FFN}
=
\operatorname{RMSNorm}(x).
\]

The router therefore has access to the same token representation the experts see.

This is the cleanest and least confounded choice.

---

# 57. Progressive sparse activation

Once the router predicts expert necessity, begin sparse execution gradually.

For eight routed experts:

```text
K=8
 ↓
K=6
 ↓
K=4
 ↓
K=3
 ↓
K=2
```

We do not need to train every intermediate K for equal durations.

The important point is avoiding an abrupt:

```text
8 → 2
```

transition.

---

# 58. Phase 2A — router-only training

Freeze:

- embeddings;
- attention/GDN;
- RMSNorm;
- FFN experts;
- LM head.

Train only:

- router parameters;
- optional expert descriptors.

This stage should be exceptionally memory-efficient.

---

# 59. Single-copy teacher trick

Because our student experts initially contain the exact dense weights, we do not necessarily need two full model copies during router-only training.

One converted model can produce:

### Dense target

```text
all experts active
```

with target tensors detached.

Then:

### Sparse prediction

```text
selected experts active
```

using the same frozen expert weights.

This reduces VRAM substantially.

Once expert weights are allowed to change, we need either:

- a separate frozen teacher;
- cached teacher targets;
- CPU/offloaded teacher weights.

---

# 60. Phase 2B — FFN recovery

After router performance stabilizes:

Unfreeze the FFN experts.

Keep:

- sequence mixer frozen;
- embeddings frozen;
- norms frozen initially.

Use a much lower LR for experts than router.

Example:

\[
LR_{\text{expert}}
=
0.05\text{–}0.1
\times
LR_{\text{router}}.
\]

Exact ratio is tuned experimentally.

---

# 61. Parameter anchor

Once experts are trainable:

\[
L_{\text{anchor}}
=
\sum_j
\|
\theta_j-\theta_j^{dense}
\|^2.
\]

This limits unnecessary drift.

We want the sparse model to adapt enough to compensate for missing computation without relearning the entire network.

---

# 62. Phase 2C — optional whole-model recovery

Only if FFN-only recovery is insufficient:

Unfreeze the whole text model with a very low LR.

This is deliberately late because full co-adaptation makes it harder to understand whether our original restructuring actually worked.

---

# 63. Distillation hierarchy

Use several levels of teacher matching.

## FFN-output distillation

\[
L_{\text{FFN}}
=
\sum_l
\frac{
\|F_l^S-F_l^T\|^2
}{
\|F_l^T\|^2+\epsilon
}.
\]

This is the most direct loss.

---

## Hidden-state distillation

\[
L_{\text{hidden}}
=
\sum_lw_l
\frac{
\|h_l^S-h_l^T\|^2
}{
\|h_l^T\|^2+\epsilon
}.
\]

This controls accumulated trajectory drift.

---

## Cosine hidden-state loss

\[
L_{\cos}
=
\sum_l
[
1-\cos(h_l^S,h_l^T)
].
\]

Optional.

---

## Logit distillation

\[
L_{\text{KD}}
=
T^2KL
\left(
P_T^T
\|
P_S^T
\right).
\]

Try:

\[
T=1
\]

and:

\[
T=2.
\]

---

## Language-model loss

\[
L_{\text{LM}}
=
-\sum_t\log p_S(x_t|x_{<t}).
\]

This keeps optimization aligned with the actual language-model objective.

---

# 64. Combined recovery objective

A practical form is:

\[
L
=
\lambda_F L_{\text{FFN}}
+
\lambda_H L_{\text{hidden}}
+
\lambda_D L_{\text{KD}}
+
\lambda_{LM}L_{\text{LM}}
+
\lambda_RL_{\text{route}}
+
\lambda_AL_{\text{anchor}}.
\]

Do not assign all terms large weights simultaneously.

---

# 65. V1 intentionally excludes variable-K

For Version 1:

\[
\boxed{K\text{ is fixed}}
\]

for every token in a given experiment.

This simplifies:

- oracle interpretation;
- routing;
- training;
- kernel execution;
- throughput analysis;
- ablations.

Variable-K becomes V2.

---

# 66. V2 variable-K concept

Later, allow:

```text
easy token
shared + E2

normal token
shared + E1 + E5

hard token
shared + E0 + E3 + E4
```

Then optimize:

\[
\min\mathbb E[K(x,l)]
\]

under quality constraints.

This is potentially powerful, but it must not contaminate V1.

---

# 67. Qwen3.5-0.8B VRAM feasibility

The official checkpoint is approximately 1.77 GB and the language model is around 0.8B parameters. citeturn571909search1turn571909search5

Therefore:

### Oracle-only phase

Comfortable under 16 GB.

No gradients required.

### Router-only phase

Also comfortable.

Almost the entire model is frozen.

### Expert-recovery phase

Still plausible under 16 GB using:

- BF16;
- gradient checkpointing;
- small microbatch;
- gradient accumulation;
- 8-bit optimizer if needed;
- teacher target caching/offload.

Actual peak VRAM must be measured rather than assumed.

---

# 68. Context length during experiments

The architecture supports much longer context, but the MoE hypothesis does not require long-context experiments initially.

Use:

```text
debug: 256–512 tokens
oracle calibration: 512–1024
router training: 1024
main recovery: 1024–2048
```

Long-context validation comes later.

This reduces:

- VRAM;
- calibration compute;
- oracle runtime.

---

# 69. Text-only scope initially

Qwen3.5-0.8B and Qwen3.8-27B are multimodal model repositories. citeturn571909search1turn480633view0

Initial research modifies only text FFNs.

For V1:

- no image calibration;
- no vision modification;
- no vision distillation.

Once text routing works, evaluate multimodal inputs because visual tokens flow through the same language-model stack.

If multimodal quality degrades disproportionately, add multimodal calibration before scale-up.

---

# 70. MTP

Qwen3.5/Qwen3.8 configurations include an MTP component. citeturn480633view1turn480633view0

Initial conversion should:

- leave MTP unchanged;
- evaluate ordinary next-token logits first;
- avoid modifying MTP internals.

MTP compatibility becomes a later validation task.

---

# 71. Phase 3 — custom Mini-Qwen3.8

We only train our own SLM **after Phase 1/2 demonstrates the core phenomenon**.

This gives us controlled science without forcing custom pretraining to become a prerequisite.

---

# 72. Why train our own model later

A custom model lets us study questions impossible with a single pretrained checkpoint:

- When does conditional FFN structure emerge?
- Does sparsifiability increase during pretraining?
- Do routing-friendly structures emerge naturally?
- Can dense training be modified to encourage later MoEfication?
- Does the architecture affect sparsifiability?

Save dense checkpoints throughout training and MoEfy each one.

That could become an interesting research result by itself.

---

# 73. Mini-Qwen3.8 architecture A — clean control

The first custom architecture can deliberately remove the hybrid sequence-mixer confounder.

```yaml
hidden_size: 768
layers: 16
intermediate_size: 2560

attention:
  type: gated_gqa
  query_heads: 12
  kv_heads: 4
  head_dim: 64

norm:
  type: rmsnorm

mlp:
  type: swiglu

vocab_size: 32768
context: 2048
```

Every layer uses gated GQA.

Purpose:

> Test whether PP-MoE works in a very conventional modern Transformer.

---

# 74. Mini-Qwen3.8 architecture B — target-faithful hybrid

Then train:

```text
Gated DeltaNet
Gated DeltaNet
Gated DeltaNet
Gated Full Attention
```

repeated four times.

This architecture mirrors the sequence-mixer rhythm used by both the Qwen3.5-0.8B text stack and Qwen3.8-27B text stack. citeturn480633view1turn480633view0

Comparing A and B answers:

> Does sequence-mixer architecture materially affect FFN sparsifiability?

---

# 75. Mini model dimensions

Recommended:

\[
d_{\text{model}}=768
\]

\[
d_{\text{ff}}=2560
\]

\[
L=16.
\]

Reason:

\[
2560
=
512+8\times256.
\]

This gives a beautiful hardware-aligned PP-MoE decomposition.

---

# 76. Mini-model shared split

```text
shared = 512

8 routed experts × 256
```

Total:

\[
512+8(256)=2560.
\]

Top-2:

\[
\frac{512+512}{2560}
=
40\%.
\]

Top-3:

\[
\frac{512+768}{2560}
=
50\%.
\]

This directly mirrors the target Qwen3.8 experiment.

---

# 77. Custom SLM pretraining budget

Do not immediately commit to a huge run.

### Smoke

\[
50\text{M–}100\text{M tokens}.
\]

### Pilot

\[
\sim500\text{M tokens}.
\]

### Useful controlled teacher

\[
2\text{B–}4\text{B tokens}.
\]

Only scale training if the existing-Qwen oracle results justify it.

---

# 78. Phase 4 — Qwen3.8-27B

The eventual target is Qwen3.8-27B.

Its official text configuration reports:

\[
d_{\text{model}}=5120
\]

\[
d_{\text{ff}}=17408
\]

\[
L=64
\]

with:

```text
linear
linear
linear
full attention
```

repeated through the text stack, plus gated attention output. citeturn480633view0

The model was released as a dense Qwen3.8 model rather than a capacity-expanded MoE. citeturn668502search5

This makes it an unusually good eventual target.

---

# 79. Candidate Qwen3.8-27B decomposition

A clean exact split is:

\[
17408
=
3072+8(1792).
\]

Thus:

```text
shared = 3072

E0 = 1792
E1 = 1792
E2 = 1792
E3 = 1792
E4 = 1792
E5 = 1792
E6 = 1792
E7 = 1792
```

All original FFN neurons are preserved exactly once.

---

# 80. Qwen3.8-27B active ratios

Top-1:

\[
\frac{3072+1792}{17408}
\approx27.94\%.
\]

Top-2:

\[
\frac{3072+3584}{17408}
\approx38.23\%.
\]

Top-3:

\[
\frac{3072+5376}{17408}
\approx48.53\%.
\]

Top-4:

\[
\frac{3072+7168}{17408}
\approx58.82\%.
\]

The most interesting eventual regimes are therefore:

\[
\boxed{38\%\text{ active FFN}}
\]

and:

\[
\boxed{49\%\text{ active FFN}}.
\]

---

# 81. Total parameter target at 27B

Partitioning does not change FFN parameter count.

So approximately:

```text
27B dense
 ↓
27B PP-MoE
 + tiny routers
```

rather than:

```text
27B dense
 ↓
100B+ conventional MoE
```

That is the defining differentiator.

---

# 82. Parameter accounting

A bias-free SwiGLU neuron contains approximately:

\[
3d_{\text{model}}
\]

parameters:

- one gate row;
- one up row;
- one down column.

Partitioning merely reorganizes these tensors.

Therefore:

\[
P_{\text{experts}}
=
P_{\text{dense FFN}}.
\]

Additional parameters are only:

- routers;
- optional expert keys;
- optional routing calibration parameters.

Target overhead:

\[
<1\%
\]

ideally.

Experimental maximum:

\[
<3\%.
\]

---

# 83. Router overhead example

For Qwen3.5-0.8B:

\[
d=1024,\quad N=8.
\]

Linear router/layer:

\[
1024\times8=8192
\]

weights.

Across 24 layers:

\[
196608
\]

router matrix parameters plus biases.

This is negligible relative to ~0.8B.

---

# 84. Physical FFN packing

After partitioning, physically reorder neurons:

```text
shared
expert 0
expert 1
...
expert 7
```

Apply the same permutation to:

```text
gate rows
up rows
down columns
```

This means the saved model is already laid out for efficient sparse kernels.

---

# 85. Research implementation vs production representation

Conceptually:

```text
Mixture of Experts
```

Implementation-wise on one GPU, the best representation may actually be:

```text
conditionally activated block-sparse FFN
```

rather than a classical distributed MoE dispatcher.

This distinction matters.

---

# 86. Why naive MoE can be slower

A dense FFN may execute:

```text
one large optimized GEMM
```

while a naive sparse implementation performs:

```text
router
scatter
small GEMM
small GEMM
gather
sum
```

Even with fewer FLOPs, the sparse implementation can lose due to:

- launch overhead;
- poor occupancy;
- memory movement;
- token dispatch;
- narrow expert matrices.

Therefore theoretical sparsity is only the first half of the project.

---

# 87. Inference implementation stages

## Kernel K0 — reference

Plain PyTorch.

Goal:

```text
correctness
```

not speed.

## Kernel K1 — packed block experts

Experts occupy contiguous tensor blocks.

## Kernel K2 — grouped GEMM

Group tokens by expert selection.

## Kernel K3 — custom block-sparse kernel

Potentially:

- Triton;
- CUTLASS;
- fused router/dispatch;
- fused expert accumulation.

---

# 88. Batch-1 decode warning

Sparse FFNs may struggle to beat dense GEMMs during autoregressive batch-1 decoding.

Therefore benchmark separately:

```text
prefill
decode
```

and:

```text
batch 1
batch 4
batch 8
batch 16+
```

A model may have:

- strong prefill improvement;
- modest decode improvement.

That is acceptable if reported clearly.

---

# 89. Evaluation hierarchy

Every major experiment needs three evaluation categories:

```text
functional fidelity
capability retention
hardware efficiency
```

---

# 90. Functional fidelity metrics

Measure:

### FFN normalized error

\[
\epsilon_l
=
\frac{
\|F_l^S-F_l^T\|
}{
\|F_l^T\|+\epsilon
}.
\]

### FFN cosine similarity

### Hidden-state MSE

### Hidden-state cosine similarity

### Logit KL

### Dense/student top-token agreement

### Perplexity

---

# 91. Error-growth curve

For every layer:

\[
D_l
=
D(h_l^S,h_l^T).
\]

Plot:

```text
layer depth
vs
student/teacher divergence
```

This makes accumulated errors directly visible.

A good sparse model should not show runaway divergence.

---

# 92. Capability benchmarks

Use a modest text benchmark set appropriate to 0.8B capability.

Examples:

- HellaSwag;
- PIQA;
- ARC-Easy;
- ARC-Challenge;
- WinoGrande;
- BoolQ;
- LAMBADA;
- OpenBookQA;
- selected MMLU categories where scores provide signal.

The principal comparison is always:

```text
same dense teacher
vs
converted sparse student
```

not external leaderboard rank.

---

# 93. Quality retention

For appropriate metrics:

\[
R_Q
=
\frac{Q_S}{Q_T}.
\]

Also report absolute difference.

Ratios can be misleading for benchmarks near chance.

---

# 94. Efficiency metrics

Report:

```text
total model parameters
router parameters
active FFN width/token
active FFN parameters/token
theoretical FFN FLOPs
estimated whole-model FLOPs
router latency
dispatch latency
expert GEMM latency
prefill tokens/sec
decode tokens/sec
peak VRAM
time to first token
```

Never report only “X% sparsity.”

---

# 95. Router diagnostics

For every layer:

- expert usage histogram;
- oracle expert usage;
- learned/oracle overlap;
- precision;
- recall;
- Jaccard similarity;
- routing accuracy where applicable;
- subset-classifier accuracy;
- expert interaction matrix;
- oracle co-selection matrix;
- average reconstruction error.

---

# 96. Token-level routing analysis

Inspect routing for:

```text
punctuation
common function words
rare words
numbers
math
code
names/entities
multilingual text
reasoning-like sequences
```

But do not force experts to become human-interpretable.

Specialization is diagnostic, not the objective.

---

# 97. V1 experiment matrix

V1 should stay intentionally narrow.

## Partition

```text
random balanced
weight based
activation based
contribution based
```

## Shared

```text
none
512-neuron shared
```

## Experts

```text
N = 8
```

Only.

## Routing budget

```text
K = 8
K = 6
K = 4
K = 3
K = 2
```

## Router

```text
linear top-K
sigmoid top-K
teacher-supervised sigmoid
subset classifier
```

That is enough for V1.

---

# 98. Things explicitly postponed from V1

Do not let these creep into the initial implementation:

```text
variable K
16+ experts
sparse attention changes
N-gram embeddings
new residual architecture
full model retraining
reinforcement-learning router
distributed expert parallelism
512 experts
dynamic expert width
```

They belong later.

---

# 99. Primary hypotheses

## H1 — exact function preservation

The partitioned all-expert FFN matches the original dense FFN.

Expected:

\[
\text{true by construction}.
\]

---

## H2 — latent conditional structure

The oracle can omit a substantial fraction of FFN blocks with limited dense-function error.

This is the central empirical hypothesis.

---

## H3 — local sparsity survives rollout

Good local FFN oracle results will remain useful when sparse-state errors propagate through the full model.

This is not guaranteed.

---

## H4 — intelligent partitioning beats random

Contribution/co-activation grouping should improve the oracle quality/compute frontier.

---

## H5 — shared neurons help

Moving ubiquitous neurons to an always-active block should improve quality at equal routed compute.

---

## H6 — necessity routing beats normalized output routing

Independent expert-necessity prediction should be better suited to additive partitioned experts than conventional softmax-normalized expert mixing.

---

## H7 — teacher supervision helps

A router trained on oracle subsets should outperform routing learned only through end-to-end LM loss.

---

## H8 — set interactions matter

If the subset classifier substantially beats independent sigmoid routing, cancellation/synergy between experts is a significant routing factor.

---

## H9 — progressive sparsification helps

Gradual:

```text
8 → 6 → 4 → 3 → 2
```

should preserve more quality than immediate:

```text
8 → 2.
```

---

## H10 — layer sparsifiability differs

Different FFN layers will have different minimum viable K.

---

## H11 — modest recovery is enough

Limited FFN adaptation should recover most sparse quality without complete model retraining.

---

## H12 — FLOP gains exceed latency gains

Sparse mathematical computation will not translate 1:1 to wall-clock speedup.

---

# 100. Go/no-go milestones

## M0 — conversion correctness

Required:

```text
all-expert converted model
≈
dense teacher
```

No meaningful perplexity difference.

Failure means implementation bug.

---

## M1 — oracle viability at 50%

At roughly:

\[
50\%
\]

active FFN, seek:

- small local reconstruction error;
- controlled rollout divergence;
- very small perplexity degradation.

A working target is:

\[
\Delta PPL\le2\%
\]

but this is a target, not an assumption.

---

## M2 — oracle viability around 35–40%

At approximately:

\[
35\%-40\%
\]

active FFN, seek:

\[
\Delta PPL\le5\%
\]

before recovery.

Again, this is an experimental target.

---

## M3 — learned router

Learned router should capture a large fraction of oracle performance.

The exact acceptance threshold should be defined after observing the oracle gap.

---

## M4 — recovery

Final student target:

\[
\text{quality retention}
\ge98\%
\]

if achievable,

with:

\[
\text{active FFN}
\le40\%-50\%.
\]

If not achievable, report the Pareto curve.

---

## M5 — real hardware gain

With optimized kernels, target:

\[
>1.2\times
\]

prefill throughput in a compute-bound regime.

Do not require the same gain for batch-1 decode.

---

# 101. Failure tree

## Oracle fails

Problem:

```text
decomposition
```

Actions:

- better neuron grouping;
- shared expert;
- fewer/larger experts;
- higher K;
- keep difficult layers dense.

Do not work on a fancy router.

---

## Local oracle good, rollout oracle bad

Problem:

```text
error accumulation
```

Actions:

- increase K in sensitive layers;
- optimize hidden-state rather than FFN-output reconstruction;
- layer-adaptive routing;
- small expert recovery;
- preserve late layers more densely.

---

## Oracle good, router bad

Problem:

```text
routing prediction
```

Actions:

- query-key gate;
- MLP router;
- subset classifier;
- ranking supervision;
- larger router input projection;
- interaction-aware loss.

---

## Independent router bad, subset classifier good

Problem:

```text
expert interactions / cancellation
```

Possible direction:

- model pairwise expert interactions;
- modify clustering to put dependent experts together;
- train set-scoring router.

---

## Shared expert dominates

If almost all compute goes through the shared subset:

- shared fraction may be too large;
- partition may not expose conditionality.

---

## Routed expert always selected

Move universally useful neurons from that expert into the shared block.

---

## Experts drift badly during recovery

Use:

- lower LR;
- stronger parameter anchor;
- freeze router temporarily;
- fewer recovery steps.

---

## Quality good, speed bad

Problem:

```text
kernel / dispatch
```

not model science.

Work on:

- packing;
- grouped GEMM;
- block-sparse kernels;
- expert granularity.

---

# 102. Phase 1 concrete first run

```yaml
experiment: qwen35_0p8b_ppmoe_oracle_v0

model:
  name: Qwen3.5-0.8B-Base
  modality: text_only
  dtype: bfloat16

conversion:
  layers: all_text_ffns
  num_experts: 8
  shared_width: 0
  expert_width: 448
  partition: random_balanced
  physical_permutation: true

oracle:
  local_teacher_forced: true
  rollout: true
  exact_subset_search: true
  k_values:
    - 8
    - 6
    - 4
    - 3
    - 2

calibration:
  initial_tokens: 100000
  context_length: 512

evaluation:
  all_expert_equivalence: mandatory
  local_reconstruction: true
  rollout_hidden_error: true
  perplexity: true
  expert_interactions: true
  co_selection: true
```

There is:

```text
no router
no distillation
no training
```

in this first run.

---

# 103. Phase 1 second run

```yaml
experiment: qwen35_0p8b_ppmoe_partition_v1

base:
  use_best_settings_from_v0: true

partition_methods:
  - random_balanced
  - weight_balanced
  - activation_balanced
  - contribution_balanced

calibration_tokens:
  - 100000
  - 500000
  - 1000000

oracle:
  exact: true
  k_values: [8, 6, 4, 3, 2]
```

Output:

```text
best partition method
+
calibration-data stability curve
```

---

# 104. Phase 1 shared-expert run

```yaml
experiment: qwen35_0p8b_ppmoe_shared_v2

ffn_width: 3584

candidate_A:
  shared: 0
  experts: 8
  expert_width: 448

candidate_B:
  shared: 512
  experts: 8
  expert_width: 384

shared_selection:
  - activation_frequency
  - contribution_frequency
  - mean_contribution
```

Choose by oracle quality at matched active compute.

---

# 105. Phase 2 router run

```yaml
experiment: qwen35_0p8b_ppmoe_router_v0

architecture:
  use_best_partition: true
  fixed_k: 3

router_variants:
  - linear_topk
  - sigmoid_topk
  - teacher_sigmoid
  - subset_classifier

train:
  base_model_frozen: true
  experts_frozen: true

loss:
  oracle_imitation: true
  ffn_reconstruction: true

evaluation:
  oracle_gap: true
  perplexity: true
```

---

# 106. Phase 2 recovery run

```yaml
experiment: qwen35_0p8b_ppmoe_recovery_v0

routing:
  best_router: true
  fixed_k: 3

trainable:
  router: true
  ffns: true
  attention: false
  norms: false
  embeddings: false

loss:
  ffn_reconstruction: true
  hidden_distillation: true
  logit_kd: true
  lm_loss: true
  parameter_anchor: true
```

Only after K=3 is stable do we attempt K=2.

---

# 107. Repository structure

```text
ppmoe/
│
├── configs/
│   ├── qwen35_0p8b/
│   ├── mini_qwen38/
│   ├── qwen38_27b/
│   └── ablations/
│
├── src/
│   │
│   ├── adapters/
│   │   ├── qwen35.py
│   │   └── qwen38.py
│   │
│   ├── moefication/
│   │   ├── neuron.py
│   │   ├── partition.py
│   │   ├── permutation.py
│   │   ├── shared.py
│   │   ├── convert.py
│   │   └── verify.py
│   │
│   ├── stats/
│   │   ├── activations.py
│   │   ├── contribution.py
│   │   ├── sketches.py
│   │   └── interactions.py
│   │
│   ├── clustering/
│   │   ├── random.py
│   │   ├── weight.py
│   │   ├── activation.py
│   │   ├── contribution.py
│   │   └── balanced.py
│   │
│   ├── oracle/
│   │   ├── exact.py
│   │   ├── gram.py
│   │   ├── rollout.py
│   │   └── metrics.py
│   │
│   ├── routing/
│   │   ├── linear.py
│   │   ├── sigmoid.py
│   │   ├── query_key.py
│   │   ├── subset.py
│   │   └── masks.py
│   │
│   ├── distill/
│   │   ├── teacher.py
│   │   ├── losses.py
│   │   └── hooks.py
│   │
│   ├── kernels/
│   │   ├── reference.py
│   │   ├── packed.py
│   │   ├── grouped.py
│   │   └── triton/
│   │
│   └── eval/
│       ├── fidelity.py
│       ├── perplexity.py
│       ├── benchmarks.py
│       ├── routing.py
│       └── efficiency.py
│
├── scripts/
│   ├── verify_conversion.py
│   ├── collect_stats.py
│   ├── cluster_neurons.py
│   ├── oracle_local.py
│   ├── oracle_rollout.py
│   ├── train_router.py
│   ├── recover_ffn.py
│   ├── evaluate.py
│   └── benchmark.py
│
└── tests/
    ├── test_neuron_triplet.py
    ├── test_permutation.py
    ├── test_ffn_equivalence.py
    ├── test_model_equivalence.py
    ├── test_oracle_bruteforce.py
    ├── test_gram_oracle.py
    └── test_bf16_drift.py
```

---

# 108. Exact implementation order

The engineering sequence should be:

```text
1. Load Qwen3.5-0.8B-Base text stack.

2. Identify actual SwiGLU module layout.

3. Write FFN neuron-triplet abstraction.

4. Write neuron permutation.

5. Test permutation equivalence.

6. Write eight-way FFN partition.

7. Test one-layer all-expert equivalence.

8. Convert every text FFN.

9. Test whole-model all-expert equivalence.

10. Implement expert-output capture.

11. Implement Gram-matrix exact oracle.

12. Verify Gram oracle against brute-force reconstruction.

13. Run 100K-token random-partition local oracle.

14. Build layer × K heatmaps.

15. Implement rollout oracle.

16. Measure PPL under rollout.

17. Decide whether the hypothesis passes the first gate.

18. Only then collect richer neuron statistics.

19. Compare intelligent partitions.

20. Introduce shared experts.

21. Finalize decomposition.

22. Train first router.

23. Measure oracle gap.

24. Progressive sparse activation.

25. FFN recovery distillation.

26. Kernel work.

27. Custom Mini-Qwen3.8.

28. Qwen3.8-27B calibration/oracle study.

29. Qwen3.8-27B routing/recovery if oracle passes.
```

---

# 109. The first graph we care about

The entire project initially boils down to one graph:

```text
y
│  dense-quality retention
│
│  ●────────●
│            \
│             ●
│              \
│               ●
│                 \
│                   ●
└──────────────────────── x
    active FFN fraction
```

We need to discover the knee of this curve.

If the knee is around:

\[
40\%-50\%
\]

active FFN, the project is highly promising.

If the knee occurs around:

\[
80\%-90\%,
\]

the achievable efficiency benefit may be too weak.

---

# 110. Second critical graph

Compare:

```text
random partition
activation partition
contribution partition
shared + contribution partition
```

on the same quality/compute plot.

This directly tells us whether our proposed functional clustering contributes anything beyond raw oracle routing.

---

# 111. Third critical graph

Plot:

```text
oracle
vs
learned router
```

If oracle is far above learned routing:

```text
routing is the research bottleneck.
```

If learned routing is close:

```text
kernel engineering becomes the bottleneck.
```

---

# 112. Main project gates

The research is now divided into four clean questions.

## Gate A — Algebra

Can the model be rewritten as experts without changing its function?

Expected:

\[
\boxed{\text{yes}}
\]

---

## Gate B — Structure

Can an oracle omit a large portion of those experts?

Unknown.

This is the central scientific hypothesis.

---

## Gate C — Prediction

Can a cheap router predict the oracle's useful expert subset?

Unknown.

This is the routing problem.

---

## Gate D — Hardware

Can the structured sparse computation actually beat dense hardware kernels?

Unknown.

This is the deployment problem.

Keeping these four gates separate is essential.

---

# 113. Why Qwen3.5-0.8B is now the correct starting SLM

It gives us:

- an already trained model;
- real language representations;
- real modern SwiGLU FFNs;
- the same broad hybrid sequence-mixer family as Qwen3.8-27B;
- small enough weights for a ≤16 GB research environment;
- no need to spend billions of tokens before seeing the first oracle result.

Therefore:

\[
\boxed{
\text{Yes, we are starting with an SLM.}
}
\]

But we are starting with a **pretrained SLM**, not by pretraining our own SLM.

That is the most important revision.

---

# 114. Why we still train Mini-Qwen later

The custom Mini-Qwen model still matters.

It becomes the controlled science environment after viability is established.

That lets us answer:

```text
When does FFN conditionality emerge?

Does it emerge before competence?

Does more pretraining make FFNs more or less sparsifiable?

Can training encourage future sparsifiability?

Does hybrid attention change the effect?

Can we design dense models specifically to be easy to MoEfy later?
```

These questions are potentially more interesting than simply obtaining one sparse checkpoint.

---

# 115. Eventual Qwen3.8-27B research strategy

Do not begin by training anything on 27B.

Start exactly as on the SLM:

```text
Qwen3.8-27B
     ↓
collect text FFN statistics
     ↓
partition selected layers
     ↓
verify all-expert equality
     ↓
oracle only
```

The public Qwen3.8-27B repository is tens of gigabytes, so this stage will require hardware/offload beyond the original single-16-GB experimental environment. citeturn571909search4

But we do not commit to expensive 27B training until the oracle study says it is worthwhile.

---

# 116. Qwen3.8-27B first large-scale milestone

Run the oracle on perhaps:

```text
4 representative layers:
early
early-middle
late-middle
late
```

before converting all 64 layers.

If the layer-level results resemble the SLM:

```text
expand to all layers
```

If not:

```text
study scaling behavior first
```

This makes scale-up incremental.

---

# 117. Important scientific comparison

Eventually compare oracle sparsifiability across:

```text
0.8B pretrained SLM
~150M custom Mini-Qwen
1–3B intermediate model
27B target
```

This answers:

> Does conditional FFN redundancy become stronger or weaker with model scale?

That may be a significant result by itself.

---

# 118. Relation to existing work

The closest conceptual precedent is **MoEfication**, which explicitly reorganizes dense Transformer FFNs into expert partitions while preserving total parameters and reported large conditional reductions in FFN usage in its evaluated models. citeturn571909academia24

Sparse Upcycling independently demonstrates the broader value of recycling dense pretrained checkpoints rather than training sparse models entirely from scratch, though conventional upcycling typically expands expert capacity rather than preserving our strict total-parameter budget. citeturn571909academia25

Our intended contribution is the combination of:

```text
modern SwiGLU LLM
+
exact parameter-preserving partition
+
oracle-first analysis
+
contribution-aware grouping
+
shared-neuron discovery
+
interaction-aware routing diagnostics
+
gated necessity prediction
+
progressive distillation
+
block-sparse deployment
+
scale study toward a 27B modern dense model
```

---

# 119. Potential novel contributions

If successful, the project could contribute several distinct findings.

### Contribution 1

Modern SwiGLU LLMs contain exploitable structured conditional redundancy inside trained dense FFNs.

### Contribution 2

Output-contribution-based clustering exposes that redundancy better than weight or activation clustering.

### Contribution 3

Shared-neuron identification improves the quality/active-compute frontier.

### Contribution 4

Independent contribution-gated routing is better suited than normalized mixture weights when experts are additive partitions.

### Contribution 5

Teacher-oracle routing provides strong supervised labels for dense→sparse conversion.

### Contribution 6

Local oracle sparsifiability and rollout sparsifiability are measurably different.

### Contribution 7

Expert cancellation/co-selection statistics explain where simple routers fail.

### Contribution 8

A block-sparse FFN implementation may be more appropriate than conventional MoE dispatch for parameter-preserving experts.

### Contribution 9

The same procedure can scale from an SLM to a 27B modern hybrid Transformer.

---

# 120. Final V1 objective

The first real target is not:

```text
build a production sparse Qwen immediately.
```

It is:

\[
\boxed{
\text{Determine whether the FFNs of Qwen3.5-0.8B contain enough structured conditional redundancy to justify the entire project.}
}
\]

The exact V1 experiment is therefore:

```text
Qwen3.5-0.8B-Base
        ↓
text FFNs only
        ↓
8 balanced partitions
        ↓
exact all-expert equivalence
        ↓
exact K-subset oracle
        ↓
rollout oracle
        ↓
perplexity / reconstruction curve
        ↓
smart clustering
        ↓
shared expert
        ↓
final oracle curve
```

No router training is necessary to reach the first major decision.

---

# 121. Desired V1 outcome

Something like:

```text
Dense teacher:
100% FFN active
PPL = baseline

Oracle PP-MoE:
50% FFN active
PPL ≈ baseline + 1–2%

Oracle PP-MoE:
~35–40% FFN active
PPL ≈ baseline + 2–5%
```

would be a very strong reason to continue.

Then Phase 2 attempts to make a learned router approach those oracle numbers.

---

# 122. Desired eventual learned SLM result

A successful first model might look approximately like:

```text
Qwen3.5-0.8B dense
100% dense FFN active

↓

parameter-preserving conversion

~same total parameters

↓

shared + routed FFNs

40–50% active FFN

↓

teacher-supervised routing
+
small recovery

↓

~98–100% dense quality retention
```

The exact numbers are experimental targets, not assumed outcomes.

---

# 123. Desired eventual Qwen3.8-27B result

The long-term target is:

```text
Qwen3.8-27B
dense FFNs
d_ff = 17408

↓

shared 3072
+
8 × 1792 experts

↓

top-3:
~48.5% active FFN

or

top-2:
~38.2% active FFN

↓

tiny router overhead

↓

near-dense quality
```

while maintaining roughly the same total parameter count rather than expanding into a very large conventional MoE.

---

# 124. Final decision

The revised strategy is:

\[
\boxed{
\text{Tiny Unit Test}
\rightarrow
\text{Qwen3.5-0.8B Oracle Study}
\rightarrow
\text{Learned Fixed-K Router}
\rightarrow
\text{Recovery}
\rightarrow
\text{Custom Mini-Qwen3.8}
\rightarrow
\text{Larger Scaling Study}
\rightarrow
\text{Qwen3.8-27B}
}
\]

The most important immediate milestone is **not training**.

It is obtaining the first reliable:

\[
\boxed{
\text{active FFN fraction}
\quad\text{vs}\quad
\text{dense-model fidelity}
}
\]

oracle curve on Qwen3.5-0.8B-Base.

If that curve is strong, we have evidence worth building on.

If it is weak, we learn that before spending significant compute.

That is the revised project plan.
