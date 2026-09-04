# The patches

日本語版: [patches.ja.md](patches.ja.md)

31 patches against llama.cpp `v0.2.0`, grouped by scope below; the application
order is the file numbering. Order matters: several touch the same files, and
later ones build on earlier ones.

Every patch carries its full reasoning — including the measurements that
justify it and the alternatives that were tried and rejected — in the comments
it adds to the source. This page is the index.

**Scope** classifies who benefits:

| Scope | Meaning |
|---|---|
| `sm_60` | Gated to Pascal GP100 (no DP4A, full-rate HFMA2). Do not apply elsewhere without measuring. |
| `pre-Volta` | Gated to NVIDIA before Volta, i.e. Pascal and older. |
| `pre-Turing` | Gated to NVIDIA before Turing. **This includes Volta, which does have tensor cores and was not measured.** |
| `all archs` | Not architecture-gated at all: every GPU gets the change. Measured only on sm_60. |
| `CUDA` | Architecture-independent CUDA improvement. |
| `CUDA (delta-net)` | Architecture-independent, but only fires for models with a gated delta-net block. |
| `host` | CPU-side; the faster the GPU, the larger the share. |
| `model` | Depends on a model architecture, not on hardware. |

Unless stated otherwise, output is **bit-identical** to the unpatched build.

---

## Pascal-specific

### 01 · `vmad-dp4a-sm60` — `sm_60`

sm_60 has no DP4A, so upstream emulates the 4×int8 dot product by extracting
bytes and multiplying. Four chained PTX `vmad` (scalar MAC with byte selectors)
do it without the extraction: ~15 → ~6.75 SASS instructions.

**+6.5–6.8% decode.** Numerically identical.

### 02 · `mmvq-rows-per-block-sm60` — `pre-Turing`

MMVQ computes one output row per block and re-reads the q8_1 activation row
from L2 once per output row. On pre-Turing NVIDIA that load — not DRAM
bandwidth (32–34% of peak) and not instruction count — is the limit (measured on
GP100/sm_60; the other pre-Turing parts are an inference from that).

Sweep at ncols_dst 1 (9B Q4_0, `llama-bench` tg32): 1 → 46.9, 2 → 54.1, **4 →
57.7 (+23.0%)**, 8 → 57.2 t/s.

**+23.0% at the shipped value of 4** on that sweep. An earlier revision that
shipped 2 rows measured +15.4% (Q4_0) / +16.5% (Q4_1) end-to-end; the +23.0%
figure is `llama-bench tg32`, which has no speculative decoding — see
[benchmarking.md](benchmarking.md) on why that harness can disagree with the
real workload. Turing and later and GCN keep upstream values.

### 05 · `mmvf-f32-pascal` — `pre-Turing`

`mmvf.cu` caps the F32 vector path at `ne11 <= 3` for pre-Turing NVIDIA, but 3
is the value used for GPUs that *have* matrix cores. Every other
no-matrix-core branch in that function returns 8, and the call site says the
vector kernel wins "for GPUs without tensor cores". Pascal has none.

Speculative decoding makes `ne11 = n_draft + 1 = 4`, one over the cap, so every
F32 `mul_mat` fell into cuBLAS SGEMM — 5.9% of decode GPU time.

**+3.7–4.3% depending on the workload**, reproduced three times independently.
The one workload whose output was checked is bit-identical (and 3.6% faster);
the others were not checked. Note the edited branch is `cc < TURING`, so it also
raises the limit for Volta, which was not measured.

### 06 · `mmq-mul-mat-id-sm60` — `sm_60`

MMQ is disabled wholesale without native DP4A. **That is correct for dense
`MUL_MAT`**: measured across every quantization tested at n=512, MMQ runs at
0.49–0.61× the cuBLAS path, and forcing it on costs a dense model 56% of its
prefill.

`MUL_MAT_ID` is different, because there the alternative is not cuBLAS but the
sorted-gather fallback: two stream syncs, a host triple loop, and one `mul_mat`
per expert. A 0.6× kernel wins easily.

**MoE prefill +20–41%** (no regression from 34 to 23,908 prompt tokens), and
~200 MiB less peak VRAM. Perplexity is statistically indistinguishable
(paired ΔNLL +0.0028 ± 0.0029, t = 0.96); against a CPU-backend reference the
patched build is *closer* (+0.038% vs +0.224%).

### 07 · `mmvq-moe-rows-sm60` — `all archs`

`rows_per_block = 2; // 2 gives best perf based on tuning` in the fused MoE
matvec launcher is hardcoded and architecture-independent, and that tuning was
not done without DP4A. The kernel is 20.1% of an MoE decode step.

**+1.9% decode**, prefill unchanged (it uses MMQ).

**This one is not architecture-gated.** The constant is a plain `constexpr`, so
every GPU gets 4 and upstream's 2 is not preserved anywhere. Only sm_60 was
measured; if you are on Turing or later, measure or revert this one.

### 08 · `mmvq-mmid-batch-sm60` — `pre-Volta`

The table gating quantized `MUL_MAT_ID` into the fused MoE matvec is keyed on
`cc < VOLTA`, lumping sm_60 in with sm_61. They differ in exactly the thing
that decides where the crossover belongs: past the cap sm_61 lands on MMQ,
which is fine, while sm_60 falls all the way to sorted-gather.

**+2.2%** (geometric mean over three prompts, no workload regressing).

Two caveats the patch states and this entry should not bury. Only IQ2_XS and
IQ3_XXS were measured directly — the whole table is raised because the reason
(on sm_60 the fallback is the slow path for every type) does not depend on the
type, which is an argument rather than a measurement, and it is the weakest link
here. And the table is keyed on `cc < VOLTA`, so the ceiling is removed for
sm_61 (P40, GTX 10xx) too, where the argument above does not hold: there the
fallback is MMQ, which is a fine place to land.

### 09 · `mmvq-nwarps-small-k-sm60` — `pre-Turing`

The block is 4 warps wide regardless of K, but a thread only enters the K loop
if `tid/(qi/vdr) < ncols_x/qk`. With small K the tail of the block idles for
the whole kernel: IQ3_XXS at K=512 used 16 of 128 threads.

Passing "just enough warps to cover K" from the host: IQ3_XXS 64.03 → 36.64 µs,
Q3_K 106.59 → 48.27, IQ2_XS 24.00 → 20.76.

**+1.29% MoE decode; dense unchanged within noise (+0.12%, floor ±0.17%).** Raising `rows_per_block` cannot fix this —
rows are iterated by the same threads, so the idle ones stay idle.

Only 1, 2 and 4 warps are instantiated, so a K that needs three of the four gets
two and each thread accumulates two terms instead of one. That is the same set
of products in a different order, so the result is correct but **not
bit-identical** there (for Q4_0, `ncols_x` in 1056..1536: `blocks_per_iter_1warp`
is 16, so `need` reaches 3 at 33 blocks; 1024 is 32 blocks, which two warps cover
exactly). When `need` is 1 or 2 the dropped warps contribute exact `+0.0f` and
the per-lane term sets are unchanged, so those stay bit-identical -- the
predicate is `need`, not the launched warp count, which is 2 for `need` 2 and 3
alike.

### 12 · `mmvq-f16-sm60` — `sm_60`

A Q4_1 GEMV built on HFMA2, the largest patch here. Per issue slot HFMA2 is
worth 4× VMAD in MACs on GP100, so the nibbles are expanded to `half2` with
LOP3 magic constants only — no integer→float conversion anywhere — and
accumulated in fp16.

**+9.4% at width 5** initially; a later revision (shape-selected warp count,
plus removing a 64-bit multiply per row caused by the 20-byte `block_q4_1`)
measured **+9.5% end-to-end** — the same within noise, but across more
workloads.

Output differs in the last bits at the widths that take this path, because the
epilogue converts once instead of twice. Perplexity at ubatch 4 moved
7.2370 → 7.2359 (−0.015%) — 200× below the ±2.9% spread of the throughput
measurement, so that **bounds** the change rather than confirming it. The
argument that it is harmless is the exactness of the dequantization (Sterbenz),
not this number.

Knobs: `GGML_A16_NCOLS_MIN` / `GGML_A16_NCOLS_MAX` restrict the width range so
the two kernels can be compared inside one binary; `GGML_A16_NWARPS`,
`GGML_A16_QVEC`, `GGML_A16_SMALL_GRID` override the derived choices.

### 31 · `fattn-f16-kv-chunk` — `pre-Turing`

The tile flash-attention kernel (the path without tensor cores) reads K and V
as F16, so with a quantized KV cache `launch_fattn` first converts the whole
cache into a scratch buffer. ggml-alloc sizes that scratch at graph reservation
for the **maximum** context: 4 bytes per token per KV column, 4 KiB/token for a
4-head × 256 KV — 320 MiB at ctx 81,920 and 1,024 MiB at 262,144, paid whether
the context is used or not, on top of the KV cache itself.

The patch converts in chunks of 16,384 tokens, runs the kernel once per chunk
with at least two parallel blocks so it writes unnormalized partials plus
(max, sum), and folds each chunk into a running accumulator with an
online-softmax merge kernel. The scratch is bounded by the chunk: 64 MiB.
`GGML_CUDA_FA_F16_KV_CHUNK` sets the chunk in tokens (rounded up to 256);
`0` disables chunking.

Measured with a 27B model (64 layers, 16 with full attention, q4_0 KV, ubatch
128, `LLAMA_DEC_SLOTS=0`), CUDA0 compute buffer:

| ctx | before | after |
|---:|---:|---:|
| 81,920 | 360 MiB | 104 MiB |
| 262,144 | 1,152 MiB | 192 MiB |

Regression check at ctx 81,920 with every layer on the GPU: prefill 11k
101.0 → 101.1, 40k 90.6 → 88.6, 66k 82.6 → 81.3 t/s; decode 20.46 → 20.39,
14.78 → 14.68, 12.42 → 12.36 t/s; peak VRAM 14,043 → 13,787 MiB, flat across
eight requests of mixed depth and length. Prefill gives up about 2% at depth to
the extra launches (13 chunks × 16 layers per ubatch at 210k); decode goes
through the vec kernel and is untouched.

What the 960 MiB buys at ctx 262,144: fitting the model needed 14 of its 48
SSM blocks (170 MiB each) on the CPU; with the patch, 8. Decode 11k 8.36 →
11.04, 40k 7.47 → 9.54, 210k 4.33 → 4.93 t/s; prefill at 210k 52.2 → 53.0 t/s;
peak 15,629 → 15,687 MiB with 582 MiB free.

`test-backend-ops -o FLASH_ATTN_EXT` passes 2,938/2,938 both at
`GGML_CUDA_FA_F16_KV_CHUNK=256` (290 cases with quantized KV and prefill-sized
batches go through the chunked path) and at the default. Output is **not
bit-identical** once the KV is longer than the chunk — the softmax is
accumulated in a different order — and identical below it, where the code path
is the original one. The mma kernel (Turing and later) uses stream-k and is
excluded: it keeps the full-size scratch.

Rejected: allocating the conversion from the pool instead of the compute
buffer. That only turns a fixed cost into one proportional to the depth in use,
and the VMM pool never shrinks, so the ceiling is the same.

Regression check on the two README models, by the procedure in
[benchmarking.md](benchmarking.md) (one binary with `GGML_CUDA_FA_F16_KV_CHUNK=0`
as the off arm, the 30-patch build as control, arms interleaved, order reversed
for the second half, ≤ 58 °C before every run, four rounds with the first
discarded). Under the README conditions (`-ngl 56 -fa 1`, F16 KV) the
conversion never runs and all three arms agree within 0.1%. With q4_0 KV at
24k depth the chunked prefill costs 0.54% (9B) / 0.30% (35B-A3B) on pp512;
decode is unchanged (vec kernel) — a −1.2% that showed up on the MoE only when
a prefill test preceded the decode test in the same process vanished with
`GGML_CUDA_DISABLE_FUSION=1` and with tg-only runs, i.e. it is the
layout-dependent fusion decision described in benchmarking.md, not the kernel.

---

## Architecture-independent CUDA

### 03 · `topk-moe-multirow` — `CUDA`

The fused MoE router kernel is restricted to a single row because its outputs
alias its logits input. It already maps `rows_per_block` (4) rows to one block
and reads each row fully before writing, so one block-wide barrier makes the
exception safe for every row a block owns.

Speculative decoding verifies `n_draft + 1` rows and hit this head-on, falling
back to a six-kernel chain (4.0% of decode GPU time versus 1.0% fused).

**+2.8–6.1% decode.**

**Output is not bit-identical above one row.** The fused kernel replaces the
argsort chain, and the two could already disagree on expert selection — removing
that mismatch is what raises the acceptance rate. The `n_rows == 1` path upstream
already fuses is untouched and stays bit-identical. No perplexity was taken for
this one.

### 04 · `concat-non-cont-flat` — `CUDA`

The non-contiguous concat kernel maps one block to each `(i1, i2, i3)` and
loops `i0` over `blockDim.x`, so only `min(ne0, 256)` threads work. Delta-net's
conv-state concat is `ne0 = 4..8` with `ne1 = 8192`: 2.1M threads to move 32k
elements, 128 KiB in 33 µs — 4 GB/s on a 732 GB/s card.

One thread per output element: **18.0 → 4.7 µs** at the real shape, 1.8% of
decode GPU time. (The 33 µs above is the same kernel measured in isolation; 128
KiB is the `ne0 = 4` end of the range, 224 KiB the `ne0 = 8` end.)

### 10 · `mmvq-q8-1-activation-cache` — `CUDA`

`ggml_cuda_mul_mat_vec_q` re-quantizes `src1` on every call, so an FFN's gate
and up matmuls quantize the same activations twice. `quantize_row_q8_1_cuda`
ignores `src0->type`, so the second result is bit-identical to the first.
34.6% of quantize launches were duplicates.

**+1.17% MoE / +0.88% dense decode** when first measured. The form shipped here
is ~0.5% slower than that figure, for the two reasons below. Neither is
optional.

**The buffer must not come from the CUDA pool.** The pool is a stack allocator
that asserts every free is the top of the stack, so an allocation held across
calls breaks it as soon as some caller allocates before `mul_mat_vec_q` and
frees after — which is exactly what `ggml_cuda_mul_mat_id`'s sorted-gather path
does. It is a plain grow-only device allocation instead. Taking it out of the
pool arena changes address layout, and `ggml_cuda_check_fusion_memory_ranges`
decides fusion from actual addresses, which is where most of the ~0.5% goes.

**The key must include `src1->data`, not just the tensor pointer.** That same
sorted-gather path declares its `ggml_tensor` inside the per-expert loop, so
every expert reuses one stack address while `data` advances. Keyed on the
pointer alone, two experts with equal token counts match and the second
silently computes against the first one's activations.

A shared buffer is also only safe on the stream that owns it, so calls on any
other stream fall back to upstream's per-call pool allocation.

Enabling upstream's MMVQ fusion on Pascal instead was measured at −0.48% and
rejected (recorded here, not in the patch): fusion removes the same duplicate
but holds two weights in one kernel, and the register increase
(Q3_K 140 → 218) costs 36% occupancy. Upstream's note that fusion is not
universally faster on Pascal holds here; this patch takes only the half that
does help.

### 14 · `getrows-narrow-rows` — `CUDA`

`get_rows` launches one block of 256 threads per gathered row. For `ne00 = 1`
that is 248,320 blocks moving one element each, 278 µs. Flattening
`(row, element)` to one dimension needs 970 blocks.

### 16 · `cpy-fastdiv` — `CUDA`

`cpy_scalar` performs six 64-bit integer divisions per element. At the
microbenchmark shape — an f32→f32 copy of 24,576 elements — it took 4.94 µs,
2.6× a trivial kernel of the same shape, and 40 GB/s, one fifteenth of measured
read bandwidth. The divisors are loop-invariant, so ggml's existing `fastdiv`
applies.

At the real shape: **11.26 → 4.96 µs per call (−56%)**, +0.82% end-to-end.

### 17 · `norm-register-cache` — `CUDA`

`rms_norm_f32` and `l2_norm_f32` read `x` twice: once for the sum of squares,
once to scale. Single-token decode gives them `grid = (1,1)` — one of 56 SMs —
at 7.79 µs per launch, 1.9% of decode. Caching the row in registers removes
the second read.

**`rms_norm<1024>` 7.79 → 6.50 µs, `l2_norm<32>` 2.96 → 2.39 µs**; +0.71%
(dense) / +0.95% (MoE).

The `rms_norm` side is gated at compile time on `block_size >= 1024`; `l2_norm`
caches at every block size. A runtime gate loses: at a 1.9 µs launch floor the
branch instructions themselves cost ~6% on the narrow norms.

### 18 · `fuse-sibling-nodes` — `CUDA`

Merges runs of identically shaped nodes into one launch. Unlike topk-moe or
rope+set_rows this eliminates **no** node, so the only requirement is that
reordering cannot change the result — checked as byte ranges at runtime.

Covers consecutive `CPY` (2.60 µs × 5 → 8.46 × 1) and `L2_NORM` (2.26 × 2 →
3.51 × 1). **+1.06%**, bit-identical.

Kill switches: `GGML_CUDA_DISABLE_FUSE_CPY`, `GGML_CUDA_DISABLE_FUSE_L2_NORM`.
`GGML_CUDA_FUSE_LOG=1` reports which rules fired; `=2` explains why they did
not.

### 19 · `fuse-pre-add-rms-norm` — `CUDA`

Upstream fuses only what comes *after* `rms_norm`; the preceding residual add
passes through. `add → rms_norm` was the largest adjacent pair at 3.11% of
decode GPU time. `rms_norm` maps one row to one block, so the add can be done
in its prologue.

`k_bin_bcast` 5,514 → 1,872 launches, at the cost of `rms_norm_f32<1024>` going
6.53 → 8.19 µs. **+0.70%**, bit-identical.
Kill switch: `GGML_CUDA_DISABLE_FUSE_PRE_ADD`.

### 20 · `fuse-add-unary-mul` — `CUDA`

`mul(unary(add(x, bias)), scale)` in one launch. Upstream's UNARY+MUL fusion
requires `ggml_are_same_shape`, which fails once `n_tokens > 1`, and never
covers the ADD.

`k_bin_bcast` 1,872 → **0** launches; total kernels 54,326 → 52,454.
**+0.72%**, bit-identical.
Kill switch: `GGML_CUDA_DISABLE_FUSE_ADD_UNARY_MUL`.

### 26 · `cpy-fused-rows` — `CUDA`

One thread per **row** instead of per element for the fused copies, when dim 0
is contiguous and the higher dimensions are dense. Index arithmetic drops from
six `fastdiv` per element to one multiply per row.

Microbenchmark at the real shape, four variants:

| | µs | GB/s |
|---|---:|---:|
| one thread per element (current) | 8.92 | 110 |
| one thread per row, `ne0` at runtime | 5.42 | 181 |
| one thread per row, `ne0` a compile-time constant | **4.66** | **211** |
| no index arithmetic at all (ceiling) | 4.17 | 236 |

The shipped code takes the third row when it can, which is the **1.9×**. **The
limit is index arithmetic, not data volume** — the ceiling is only 12% further
on.
Kill switch: `GGML_CUDA_DISABLE_CPY_ROWS`.

### 28 · `top-k-partial` — `CUDA`

`ggml_top_k` uses `cub::DeviceTopK::MaxPairs`, which **requires CCCL ≥ 3.2**.
CCCL 3.x ships with CUDA 13, so on **every CUDA 12.x release** it is
unavailable and the code falls back to
`argsort_f32_i32_cuda_cub` and fully radix-sorts the vocabulary to take the
first *k*. With speculative decoding this runs once per token with k=10 over
151,936 entries: ~172 µs, 1.1% of decode.

Two-stage partial selection instead. Comparisons use the same monotonic uint32
mapping cub uses, with the key in the high 32 bits and `~idx` in the low 32, so
a plain unsigned compare reproduces cub's stable order exactly — including
signed zeros and NaNs — and the result is bit-identical.

Microbenchmark at n=151,936: cub full sort 157.2 µs vs **20.6 µs at k=10
(7.6×)**, with zero mismatches against the full sort over a set seeded with 40
ties and a −inf.

Two things made it fast, both measured: replacing the block-wide tree reduction
with warp shuffles (70.9 → 42.1 µs — the *k* rounds of barriers were the
limit), and packing into 64 bits for one branch-free shuffle (42.1 → 20.6). The
elements-per-thread count barely matters: it is latency-bound per round, not
work-bound.

Falls back to the existing sort for k > 16 (a hard implementation cap — the
candidates live in per-thread register arrays), for narrow rows, and for
oversized grids.
Kill switch: `GGML_CUDA_DISABLE_TOP_K_PARTIAL`.

> This one is worth attention beyond Pascal: **every CUDA 12.x user with GPU
> sampling pays a full-vocabulary sort per token.**

### 29 · `mmvq-iq3xxs-grid-smem` — `CUDA`

`vec_dot_iq3_xxs_q8_1` reads `iq3xxs_grid` once per four weights, and the 32
indices a warp holds are unrelated, so one load is broken into as many sectors
as there are distinct addresses and replayed serially. An instruction count
hides this: the loads are a handful of the ~946 instructions in the loop body,
so counting them predicts a few percent where the measurement shows 40%.

Staging the 1 KiB table in shared memory at the top of the kernel removes it.

Isolated GEMV, K=5120, N=17408, only the lookup changed, µs/Mweight:

| grid lives in | nc=1 | nc=2 | nc=3 |
|---|---:|---:|---:|
| global (stock) | 2.965 | 4.103 | 5.204 |
| `__constant__` | 6.486 | 6.484 | 6.516 |
| **shared** | **1.824** | **2.972** | **4.000** |
| uniform index (control, wrong results) | 1.842 | 2.944 | 4.013 |

Shared memory reaches the control: the lookup cost goes to zero rather than
down. `__constant__` is 2.2× *slower* than stock, because the constant cache
broadcasts one address per cycle and a divergent index serialises 32 ways.

On a 27B dense model whose IQ3_XXS tensors carry 27% of the weights, with
MTP speculative decoding on and greedy sampling (temperature 0, top-k 1):

| prompt | stock | patched | |
|---|---:|---:|---:|
| short (256 tokens generated, two rounds) | 27.03 / 27.04 | 29.08 / 29.10 | **+7.6%** |
| 33k tokens | 18.70 | 19.80 | **+5.9%** |
| 60k tokens | 15.51 | 15.99 | **+3.1%** |

The gain shrinks with context because attention and the KV-to-f16 expansion
take a growing share of the pass, leaving `mul_mat_vec_q` a smaller one.

**Output is bit-identical** at every point: same generated-text hash, same
accepted-draft count, same pass count, so there is no acceptance-rate term to
confound the comparison. Peak VRAM is unchanged to the MiB.

Registers fall too, since the grid address arithmetic disappears: the `nc=1`
kernel goes 94 → 80, and shared memory per block grows by exactly 1024 bytes.

`iq3xxs_grid_smem_init()` must be called before any early return, since it
contains a `__syncthreads()`.

The same divergent lookup exists in MMVQ for `iq2xxs_grid`, `iq2xs_grid`,
`iq2s_grid`, `iq3s_grid` and `iq1s_grid_gpu`. None are touched here: those
tables run from 2 KiB to 8 KiB, so what they would cost in occupancy is a
different trade. MMQ reads this same 1 KiB table in `load_tiles_iq3_xxs`, which
is not touched either.

`mul_mat_vec_q_moe` gets the same initialisation because it shares the
`vec_dot`. That kernel only runs for MoE `MUL_MAT_ID` with more than one token,
which a dense model never reaches, so it was measured on its own: a 35B-A3B MoE
with 31.7% of its bytes in IQ3_XXS went 100.40 / 101.27 → 101.83 / 102.00 t/s
against the 28-patch build, two rounds with the order alternated. MTP
speculative decoding was on — 124 of 148 drafts accepted, verify width 2.12 —
which is what puts more than one token through `MUL_MAT_ID` and reaches the
kernel at all, and all four runs agree on the hash, the accepted-draft count
and the pass count, so acceptance rate does not enter this comparison either.
VRAM is unchanged. At +1.1% the effect is far smaller than the dense figures
above, so take it as indicative; what it settles is that the added barrier and
the 1 KiB fill cost nothing measurable on that path.

### 30 · `mmvq-ksigns-smem` — `CUDA`

The `vec_dot` for IQ2_XXS, IQ2_XS and IQ3_XXS turns a packed 7-bit sign field
into two per-byte masks and applies them:

```c
const uint32_t signs = unpack_ksigns(aux32 >> (7*l0/2));   // s * 0x01010101
const int signs0 = __vcmpne4(signs & 0x08040201, 0);
const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
```

Every step of that is emulated on sm_60. There is no byte-wise SIMD compare and
no byte-wise SIMD subtract, so `__vcmpne4` and `__vsub4` each expand into several
`LOP3`/`LOP32I`, and the `s * 0x01010101` broadcast is a 32-bit integer multiply,
three `XMAD`. The IQ3_XXS loop body is 120 `LOP32I` and 112 `LOP3` per 32
weights, and nearly all of it is this.

Deleting the sign application outright — wrong results, as a ceiling probe — is
**−30.4%** on the isolated GEMV at K=5120, N=17408, `nc=1`. It was the largest
single item left in that kernel after patch 29, ahead of the shared-memory grid
lookup (12.5%), the dp4a emulation (8.1%) and the activation loads (4.1%).

The masks depend only on the 7 bits they come from, so a 128-entry table filled
once per block replaces the whole chain with one shared-memory load. Two facts
keep it small and the application cheap:

- The 8th sign is the parity of the other seven, so 128 entries cover it. The
  garbage bit the stock code tolerates in the byte it truncates cancels the same
  way it does there.
- Every byte of `iq2xxs_grid`, `iq2xs_grid` and `iq3xxs_grid` is at least 4, so
  `(g ^ 0xff) + 1` per byte cannot carry into the next byte, and a plain 32-bit
  add replaces `__vsub4`.

Layout matters more than it looks. Three were measured against stock on the
isolated IQ3_XXS GEMV at K=5120, N=17408, and all three are bit-identical to it:

| table layout | `nc=1` | `nc=2` | `nc=3` |
|---|---:|---:|---:|
| stock | 149.5 µs | 198.2 µs | 257.3 µs |
| `uint4`, one 128-bit load | 130.9 | 202.3 | 247.5 |
| `uint2`, one 64-bit load | 125.2 | 187.8 | 244.4 |
| **two `uint32` from a flat 256-entry array** | **128.4** | **188.9** | **236.0** |
| ceiling: no sign application (wrong results) | 103.9 | 153.1 | 214.0 |

Keeping the stock chain and replacing only the multiply with `__byte_perm` is
worth −1.9% / −2.6%, so the multiply is not where the cost is.

`test-backend-ops -o MUL_MAT perf`, m=4096 k=14336, same flake for both arms,
interleaved with the order reversed for the second round:

| type | n=1 | n=2 | n=3 | n=4 |
|---|---:|---:|---:|---:|
| IQ3_XXS | **−21.1%** | **−9.5%** | **−9.3%** | **−8.7%** |
| IQ2_XXS | −3.7% | −7.4% | −8.6% | −8.0% |
| IQ2_XS | +0.3% | −9.2% | −7.2% | −7.2% |
| IQ4_XS (control) | −0.2% | +0.1% | −0.0% | −0.1% |

At `n=1` IQ3_XXS goes 99.65 → 78.62 µs and lands on IQ4_XS (78.44) and Q4_1
(78.65): the type that costs 3.06 bits per weight now costs the same time per
weight as the two that cost 4.25 and 5.00.

IQ2_S is left alone. It packs its signs as two nibbles rather than a 7-bit
field, and the low half of this table would serve it, but routing it through
there regressed `n=2` by 6.5% across two rounds while helping `n=3` and `n=4`.

`-o MUL_MAT` 1134/1134 and `-o MUL_MAT_ID` 790/790 pass.

On a 27B dense model with 30% of its GPU-resident bytes in IQ3_XXS, MTP
speculative decoding on and greedy sampling (temperature 0, top-k 1 — which is
what makes the bit-identical check below possible), three rounds short / two
rounds long with the order alternated:

| prompt | 29 patches | +30 | |
|---|---:|---:|---:|
| short (256 generated) | 28.92 / 28.90 / 28.87 t/s | 29.43 / 29.40 / 29.43 | **+1.8%** |
| 33k tokens | 19.97 / 19.86 | 19.92 / 19.98 | +0.2% |
| 60k tokens | 16.32 / 16.13 | 16.35 / 16.48 | +1.2% |

Output is bit-identical at every point — same generated-text hash, same
accepted-draft count, same pass count — so no acceptance-rate term enters the
comparison. Peak VRAM is unchanged to the MiB. Shared memory per block grows by
1 KiB.

The end-to-end figure is far smaller than the kernel figure, and 0.2–1.2% at
long context, for the same reason as patch 29: attention and the KV-to-f16
expansion take a growing share of a long-context pass.

A 35B-A3B MoE carrying 47.9% IQ2_XS and 31.7% IQ3_XXS, which reaches the
`mul_mat_vec_q_moe` kernel rather than `mul_mat_vec_q`, went 101.79 / 101.83 →
103.74 / 102.83 t/s over two rounds with the order alternated — **+1.5%**, or
19.05 → 18.78 ms per verify pass. Hash, accepted-draft count, pass count, verify
width and VRAM all agree across the four runs. The gain is modest for a model
that carries 80% of its bytes in these two types because only a few experts are
read per token.

`ksigns_smem_init()` must be called before any early return, since it contains a
`__syncthreads()`. MMQ reads the same sign encoding in `load_tiles_iq2_xxs`,
`load_tiles_iq2_xs` and `load_tiles_iq3_xxs`, and is not touched.

> A harness note worth more than the patch. The first version of this
> measurement used a stand-alone GEMV written with one output row per block. It
> put the sign path at 4% of the kernel and the table at 3% *slower*.
> `mul_mat_vec_q` uses four rows per block (patch 02), which changes what the
> kernel is bound by; rewritten to match, the same two measurements read 30% and
> 14% *faster*. A microbenchmark that does not reproduce the real kernel's
> launch geometry can invert the sign of a result, not merely blunt it.

---

## Delta-net / gated-delta-net (Qwen3-Next, Qwen3.5)

These are architecture-independent CUDA changes, but they only fire for models
with a gated delta-net linear-attention block.

### 23 · `fuse-gdn-beta-sigmoid` — `CUDA`

`beta = sigmoid(x)` folded into `gated_delta_net`. `unary_op_kernel<op_sigmoid>`
fired as often as `gated_delta_net_cuda` itself (4,080 launches, 6.86 ms), and
4,032 of those had `grid = 1` — pure launch overhead. gdn already folds the `g`
side's `expf`.

Node-preserving: gdn also writes the folded node's dst, so no use-count
analysis is needed and the result is bit-identical.
Kill switch: `GGML_CUDA_DISABLE_FUSE_GDN_BETA`.

### 24 · `fuse-gdn-state-gather` — `CUDA`

The SSM state `GET_ROWS` (a 2 MiB gather) folded into `gated_delta_net`.
`k_get_rows_float_vec` fired as often as gdn (3,984 launches, 46.24 ms, 1.4% of
decode) and did nothing but copy 2 MiB out of the cache and hand it over.
gdn already writes the *new* state directly into the cache, so it reads the
input directly too.

The `GET_ROWS` is ten nodes ahead, so this adds a mechanism to defer a node's
execution and fold it into its consumer. Disabled when concurrent streams are
in use, where reordering is unsafe.
Kill switch: `GGML_CUDA_DISABLE_FUSE_GDN_GATHER`.

### 25 · `gdn-gather-single-snapshot` — `CUDA`

Snapshot the row indices once per graph rather than once per layer. Every SSM
layer uses the same index tensor; doing it per layer added 4,080 device-to-device
copies (+7.31 ms).

### 27 · `fuse-concat-gather` — `CUDA`

The conv-state `GET_ROWS` folded into the `CONCAT` that follows it, which
requires making `dim == 0` CONCAT run one thread per row. `build_conv_state`
gathers a 96 KiB row and immediately concatenates it; both kernels ran at 42 and
51 GB/s, 7–8% of the ceiling, limited by index arithmetic (`concat_cont<T,0>`
does a **64-bit runtime division** by an `ne0` of 4, and GP100 has no 64-bit
divider).

Microbenchmark: 7.203 µs for the two kernels → **2.819 fused (−61%)**.

**Per-row alone is a null result** (+0.006% mean, 3 of 8 rounds positive):
one thread per row only wins when the dst row is exactly 16 B — one vector
store. Per-row **plus** fusion is +0.74% mean, 7 of 8 rounds positive, because
a whole launch disappears. The non-fused path is therefore restricted to
16 B rows.

Kill switches: `GGML_CUDA_DISABLE_CONCAT_ROWS`,
`GGML_CUDA_DISABLE_FUSE_CONCAT_GATHER`.

---

## Host side

### 11 · `penalties-direct` — `host`

`llama_sampler_penalties_apply` probes an `unordered_map` once per candidate,
but the map holds at most `penalty_last_n` entries (64 by default) while the
candidate list is the whole vocabulary. At n_vocab 151,936 that is one hash
lookup per candidate of which at most 64 can hit — roughly one in 2,400 does
any work. **0.78 ms per token, 5.3% of throughput.**

Walk the penalised tokens instead, after verifying (at most 64 comparisons)
that `cur_p->data[id].id == id` for each. Falls back to the original scan if
any check fails. The same arithmetic runs in the same order per token, so the
output is bit-identical.

> Also worth attention beyond Pascal: this is host-side, so its share *grows*
> as the GPU gets faster, and vocabularies keep growing (Gemma 3 is 262,144).

### 13 · `sampler-prefilter` — `host`

`common_sampler::set_logits` writes a `llama_token_data` per vocabulary entry
every token (12 B × 151,936 = 1.82 MiB), and the first order-sensitive sampler
reads all of it back. It runs after `llama_synchronize()`, so the GPU is idle
throughout: **2.2% of decode, entirely on the critical path.**

Do not build what top-k discards: one pass over the raw logits keeping the top
`top_k + penalty_last_n`. Penalties only ever *lower* logits and touch at most
`penalty_last_n` entries, so that set provably contains the true post-penalty
top-k. The scan tests a whole 64-element block with an **OR of comparisons**, so nearly
every block is discarded by one vectorized pass. A max reduction here is 8×
slower — `std::max` is not associative across NaN, so the compiler will not
vectorise it and it becomes a serial dependency chain: 130 µs against 19 µs per
token.

Falls back to the full build for non-equivalent chains (grammar, logit_bias,
mirostat, DRY, top-n-sigma, an unknown sampler before top-k).
Kill switch: `LLAMA_SAMPLER_PREFILTER=0`.

### 21 · `sched-reset-lazy` — `host`

`ggml_backend_sched_reset` memsets the whole hash table. It is sized from
`graph_size` — tens of thousands of entries — while one graph fills a few
thousand, so this moved over 1 MiB per graph. Measured against the alternatives
in the same build: **`sched_reset` 94 ms (118 µs/call) vs `alloc_graph` 58 ms
vs `build_graph` 8.5–54.5 ms — the reset was the largest term.**

Walk the used bitmap and restore only touched entries. The invariant that
unused entries are `(-1, NULL)` is established once in
`ggml_backend_sched_new` and maintained inductively.

**+0.94%**, bit-identical.

> Not Pascal-specific and not speculative-decoding-specific: this is a general
> llama.cpp improvement.

### 22 · `decode-sched-slots` — `host`

Upstream keeps one graph slot, so a workload whose graph type and `n_tokens`
change every step reuses it only ~14% of the time (140 hits per 1000 graphs).
Adding slots does not help while there is a single scheduler — allocation has to
be redone, and the value **per hit** drops 15×: an experiment that pushed reuse
to 88% still gained only +0.21%.

Giving each slot its own scheduler restores the real thing: skip build, reset,
split and alloc entirely.

| Slots | True reuse (hits / 1000 graphs) | Dense A/B | VRAM |
|---:|---:|---|---:|
| 0 (upstream) | 140 | — | baseline |
| **4** (default) | 275 | **+0.97%** | +72 MiB |
| 8 | 295 | +1.35% | +132 MiB |
| 16 | 301 | — | — |

Slots are skipped entirely when the context carries backend samplers. A sampler
object is shared across slots but caches `ggml_tensor` pointers into the graph it
was last applied to (`penalties`' `inp_token_ids` / `inp_counts`, `dist`'s
`inp_uniforms`), and `set_input` writes through them on every decode including a
slot hit — so alternating slots would write one slot's inputs while another
slot's graph runs.

The token cap is 4 because slot buffers are laid out differently from the main
scheduler's, and `ggml_cuda_check_fusion_memory_ranges` decides fusion from the
actual tensor addresses. topk-moe's exception (`ggml_nrows(node) <=
GGML_CUDA_TOPK_MOE_ROWS_PER_BLOCK`, from patch 03) is why 5 is where the MoE
flips — a verify batch of `n_draft + 1` falls just outside it — but it covers
only the topk-moe call sites, so it explains the boundary rather than proving the
bound. The bound itself is measured: at `n_tokens <= 4` both a dense and an MoE
model stayed bit-identical.

4 is the default because the asymmetry is severe: running out of VRAM means the
model does not load, while 0.38 points — real, and well above the ±0.07%
resolution established in [benchmarking.md](benchmarking.md), but small — is not
worth that risk. Raise it with `LLAMA_DEC_SLOTS=8` where there is headroom; `=0`
restores upstream behaviour.

`LLAMA_DEC_MAX_TOK` defaults to 4, and that number is load-bearing:
`ggml_cuda_check_fusion_memory_ranges` decides fusion from actual tensor
addresses, so a different buffer layout changes which fusions fire and
therefore changes the output. topk-moe skips that check at
`nrows <= GGML_CUDA_TOPK_MOE_ROWS_PER_BLOCK` (= 4), and inside that bound the
layout does not matter — measured bit-identical at `n_tokens <= 4`, changing at
5 and above.

---

## Model-specific

### 15 · `mtp-draft-vocab` — `model`

Restricts the vocabulary of the **MTP draft head only**. Splitting a width-1
Q4_1 GEMV by grid X shows 12.7% of it is one tensor — `output.weight`, 606 MiB
— at 1188.8 µs per call, 535 GB/s, 73% of HBM2 peak: purely bandwidth-bound.
75% of that is the draft side, 8.1% of the decode wall.

The draft only needs its own argmax and probability, and **every proposal is
verified by the target**, so shrinking the vocabulary cannot change which tokens
the target accepts — only how often a draft is accepted. The demonstration pins
the draft width so both builds share an FP path; at the default `p-min` the
*number* of drafted tokens changes, so byte-identical output is not claimed
there. Demonstrated: with `--spec-draft-p-min 0` (fixed
width, identical FP path) all three prompts hash identically with and without
the subset; degrading the subset to 277 tokens drops acceptance to 0.01 while
the output hash stays intact.

**The gain is set by coverage, and goes negative when coverage breaks:**

| Coverage | Effect |
|---:|---:|
| 97.5% | +8.3% |
| 92.8% | +5.9% |
| 90.3% | +3.9% |
| 54.2% | −3.3% |
| 33.5% | **−17.4%** |
| 29.7% | −14.6% |

Build the subset by including top BPE ranks per script unconditionally —
corpus-frequency order alone falls off a cliff on any language missing from the
corpus. The downside has an amplifier: the softmax denominator only covers the
subset, so the draft looks overconfident, passes the p-min gate more often, and
produces *more* wrong drafts.

Off unless `LLAMA_MTP_DRAFT_VOCAB` points at a token list.

Patch 14 (`getrows-narrow-rows`) is a prerequisite: the expansion back to
`[n_vocab]` is a `get_rows` with `ne00 = 1`, which cost 278 µs and turned the
whole thing **−29%** until that was fixed.
