# The patches

日本語版: [patches.ja.md](patches.ja.md)

45 patches, grouped by scope below; the application order is the file
numbering. Order matters: several touch the same files, and later ones build
on earlier ones. All 45 are generated against llama.cpp `v0.5.0`, where they
apply at zero fuzz and zero offset (see [patches.nix](../nix/patches.nix)
and the top-level README for the current `llamaCppTag`).

Patch 43 `fuse-hc-combine` was deleted at `v0.5.0`. Upstream added dedicated
`GGML_OP_DSV4_HC_PRE` / `DSV4_HC_POST` ops for hyper-connections and enabled
them by default in the qwen4exp graph (ggml-org/llama.cpp#28901). That removed the op sequence 43
matched from every layer; only the final head mix (`il = -1`) still takes the
unfused path, once per graph, which is not worth a kernel of its own. The numbering keeps the gap.

Every patch carries its full reasoning — including the measurements that
justify it and the alternatives that were tried and rejected — in the comments
it adds to the source. This page is the index. Patches 32–48 are the exception:
their measurements live on this page rather than in the source.

**Scope** classifies who benefits:

| Scope | Meaning |
|---|---|
| `sm_60` | Gated to Pascal GP100 (no DP4A, full-rate HFMA2). Do not apply elsewhere without measuring. |
| `pre-Volta` | Gated to NVIDIA before Volta, i.e. Pascal and older. |
| `no DP4A` | Reached only by NVIDIA GPUs without DP4A — GP100 and Maxwell/Kepler — which take their own MMQ tile table. A build compiled only for such an arch also routes newer cards here. |
| `pre-Turing` | Gated to NVIDIA before Turing. **This includes Volta, which does have tensor cores and was not measured.** |
| `all archs` | Not architecture-gated at all: every GPU gets the change. Measured only on sm_60. |
| `CUDA` | Architecture-independent CUDA improvement. |
| `CUDA (delta-net)` | Architecture-independent, but only fires for models with a gated delta-net block (Qwen3-Next, Qwen3.5, Qwen3.8-Flash-Next and the other architectures in the same filter). |
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

### 40 · `mmq-iq4-nl-threads` — `no DP4A`

`ggml_cuda_mmq_get_config_pascal_older` — the table for GPUs with no DP4A — is
uniform at 256 threads per block across quantization types, and that uniformity
was never measured on a device that has to emulate the int8 dot product. IQ4_NL
at 128 threads is **3–4% faster for an MoE down projection ([640, 2560], IQ4_NL)
at every token count measured**, and the kernel's error against a CPU F32
reference came out **bit-identical** to the 256-thread build in every shape
measured (measured, not argued from the tile geometry).

IQ3_XXS was swept the same way and keeps 256. The other tables (DP4A Pascal,
Ampere, Blackwell, CDNA, RDNA) are untouched; note that
`ggml_cuda_highest_compiled_arch` decides which table a card takes, so a build
compiled only for a no-DP4A arch sends every card here.

## Architecture-independent CUDA

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

Since `v0.5.0` upstream normalizes the gated delta-net q/k with `rms_norm` +
`scale` instead of `ggml_l2_norm` (ggml-org/llama.cpp#28068), so the `l2_norm` half no longer
fires on qwen35, qwen35moe, qwen3next, qwen4exp and the like; only rwkv7 still
uses `ggml_l2_norm`.

### 18 · `fuse-sibling-nodes` — `CUDA`

Merges runs of identically shaped nodes into one launch. Unlike topk-moe or
rope+set_rows this eliminates **no** node, so the only requirement is that
reordering cannot change the result — checked as byte ranges at runtime.

Covers consecutive `CPY` (2.60 µs × 5 → 8.46 × 1) and `L2_NORM` (2.26 × 2 →
3.51 × 1). **+1.06%**, bit-identical.

Kill switches: `GGML_CUDA_DISABLE_FUSE_CPY`, `GGML_CUDA_DISABLE_FUSE_L2_NORM`.
`GGML_CUDA_FUSE_LOG=1` reports which rules fired; `=2` explains why they did
not.

Since `v0.5.0` the `L2_NORM` sibling fusion no longer fires on gated delta-net
models, for the reason given under patch 17; the `CPY` one is unaffected.

### 19 · `fuse-pre-add-rms-norm` — `CUDA`

Upstream fuses only what comes *after* `rms_norm`; the preceding residual add
passes through. `add → rms_norm` was the largest adjacent pair at 3.11% of
decode GPU time. `rms_norm` maps one row to one block, so the add can be done
in its prologue.

`k_bin_bcast` 5,514 → 1,872 launches, at the cost of `rms_norm_f32<1024>` going
6.53 → 8.19 µs. **+0.70%**, bit-identical (measured on a dense model).

**Off by default since the v0.4.0 rebase.** See the hazard note after patch
20 below for why, and what enabling it (`GGML_CUDA_FUSE_PRE_ADD=1`) actually
does and does not guarantee.

### 20 · `fuse-add-unary-mul` — `CUDA`

`mul(unary(add(x, bias)), scale)` in one launch. Upstream's UNARY+MUL fusion
requires `ggml_are_same_shape`, which fails once `n_tokens > 1`, and never
covers the ADD.

`k_bin_bcast` 1,872 → **0** launches; total kernels 54,326 → 52,454.
**+0.72%**, bit-identical. Off by default, same as 19 — see below.

**Known hazard, neither 19 nor 20 goes through
`ggml_cuda_check_fusion_memory_ranges`; both are opt-in.** Both fusions add a
write the unfused graph does not have (the ADD's `dst`), and both read the
prologue's inputs (`add->src[0]`/`src[1]`, for 19; the bias add's inputs, for
20) in the same kernel that writes the epilogue's output (the `MUL` node's
`dst`). `add->src[*]` (19) goes out of scope at the ADD node, so once
ggml-alloc carves the `MUL` output out of that now-freed block, a later
allocation sharing part of that block can have its epilogue write land on
memory the prologue of a *different* invocation still needs to read — a
partial overlap upstream's check exists to catch (unfused, the three kernels
run sequentially, so this is safe). It went unnoticed through `v0.2.0` and
surfaced as non-deterministic decode output on `v0.4.0`'s MoE graphs (same
seed, three runs, three different outputs), even though the fused kernel is
correct in isolation.

The first fix tried was gating 19 off whenever the graph contains a
`MUL_MAT_ID` node (i.e. is MoE). That is not sufficient: `ggml_backend_sched`
can split a graph across backends (`--n-cpu-moe`, `-ot "exps=CPU"`), and a
`MUL_MAT_ID` routed to a CPU split is invisible to a CUDA-side scan of the
same graph, so the guard reads "dense" for a graph that is not. 20 never had
this guard at all, and duplicates the same read-under-write shape for the
gated delta-net's gate preprocessing — the architecture the non-determinism
was actually observed on. Given that, both are now **off by default**
(`GGML_CUDA_FUSE_PRE_ADD=1` / `GGML_CUDA_FUSE_ADD_UNARY_MUL=1` to opt in) —
correct on a dense model (measured bit-identical), but not guaranteed safe on
any graph the scheduler may split. The actual fix belongs upstream's own
memory-range check, not a per-patch workaround. Note that
`ggml_cuda_check_fusion_memory_ranges` itself only accepts a contiguous
`(node_idx, node_count)` range, while `ggml_cuda_collect_ops` can return
non-contiguous `idxs` — calling the check as-is would not close this on its
own.

Enable switches (default off, unlike every other patch's kill switch):
`GGML_CUDA_FUSE_PRE_ADD=1` (19), `GGML_CUDA_FUSE_ADD_UNARY_MUL=1` (20).

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

CUDA-only: the whole block is guarded with `#if !defined(GGML_USE_HIP)`. Two
reasons. `__shfl_xor_sync` here uses cub's 3-argument form, but
`ggml-cuda/vendors/hip.h` `#define`s it as a 4-argument macro, so it fails to
compile under HIP as written. And even past that, the kernels assume a
32-lane warp throughout (`CUDA_TOP_K_WARPS`, `threadIdx.x & 31`/`>> 5`,
`e*32`), which does not hold on AMD's 64-lane wavefronts. Since `v0.4.0`,
upstream ships its own HIP-only fallback, `top_k_radix_cuda`
(`!GGML_CUDA_USE_CUB && GGML_USE_HIP`, `ncols > 1024`), and this patch leaves
that path — and the full sort below `ncols = 1024` — untouched on HIP.

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

### 35 · `mmvq-mmid-batch-cap` — `CUDA`

`MUL_MAT_ID` only takes MMVQ up to `MMVQ_MAX_BATCH_SIZE` = 8 columns; wider
batches go to MMQ, which needs DP4A, or to cuBLAS. The kernel itself has no
such limit — the token index lives in `threadIdx.y` and the real constraint is
`warp_size × ncols_dst <= 1024`. 8 is where Pascal measured MMQ to win, not a
correctness bound. MMQ also cannot take a repeated id in one token
(`mm_ids_helper` writes out of bounds), which the expert cache in patch 46 does
deliberately, so that path must stay on MMVQ.

`LLAMA_MMVQ_MMID_MAX=<n>` raises the cap to at most 32, and never past
`1024/warp_size` since one block is `warp_size × ncols_dst` threads (unset keeps
upstream behaviour, including the per-type tables this override otherwise
bypasses on every architecture). `__launch_bounds__` is compile time, so a wider batch needs its own
template instantiation; the default one is left alone to keep its register
budget.

This is what makes a resident expert cache survive speculative decoding: a
verify batch is 1 + draft tokens, so every verify step exceeded the cap and ran
without the cache. Cost of one ubatch of *n* tokens at 40k / 210k depth, cache
off vs on with the cap raised (minimum of three, slots restored from the same
state):

| n | 40k off | 40k on | 210k off | 210k on |
|---:|---:|---:|---:|---:|
| 9 | 583.6 ms | **533.5** | 774.0 | **712.9** |
| 17 | 844.8 | **814.4** | 1120.8 | **1013.0** |
| 32 | 1187.8 | **1106.7** | 1425.2 | **1371.3** |

**−3.6 to −9.6% across the three widths shown**, and −3.6 to −11.2% across the
full eight-width sweep those three are taken from. VRAM identical. Greedy decode at matched
acceptance (1.000) went 44.98 / 45.69 → 43.28 / 43.88 ms/token. A 26-question ×
3-depth long-context benchmark scored the same as the capped build (61/78,
McNemar p = 1.000 on every comparison).

Throughput in t/s cannot be used for this comparison at all: the expert cache
carries over between requests, so the same prompt takes a slightly different
numerical path, which changes the generated tokens and the draft acceptance
rate. The same arm varied 13.5–23.1 t/s until acceptance was held fixed.

### 36 · `mmvq-chunk-large-batch` — `CUDA`

A quantized `mul_mat` with 8 < ne11 <= 32 falls past MMVQ and MMQ into cuBLAS,
which wants the whole weight dequantized to F16 in the pool — 1,212.5 MiB for a
248,320 × 2,560 Q6_K LM head — and the pool never shrinks, so **one speculative
verify batch keeps that resident for the rest of the process**. Cutting src1 and
dst into 8-column views and looping MMVQ needs no copy at all.

`GGML_CUDA_MMVQ_CHUNK_MIN_MIB=<MiB>` is the minimum F16 expansion that makes
the split worthwhile; unset keeps the cuBLAS path. At 256 it takes the LM head
and nothing else. The path is also gated on `ggml_cuda_should_use_mmvq`. That is belt-and-braces
rather than a type whitelist: it keeps the chunking off where upstream's own
heuristic would not have used MMVQ at this width, and on the architecture measured
here it is always true. A quantized type with no MMVQ instantiation would still
reach the switch's `GGML_ABORT`; none is reachable today.

| | pool peak | LM head request | 262k speculative decode |
|---|---:|---:|---:|
| before | 1,228 MiB | 1,212.5 MiB | 38.5 ms/token |
| after | **62 MiB** | **0** | **37.3 ms/token** |

In the production tree, this change plus the 12 extra resident expert slots the
freed VRAM buys took decode at 40k from 47.30 to 45.07 ms (−4.7%, three
interleaved rounds); the two were not separated.

Output changes: the verify logits now go down the same MMVQ path as
non-speculative decode instead of cuBLAS F16, so speculative and plain decode
agree where they did not before. Three real tasks produced **identical draft
counts, acceptance counts and generation lengths** (2,080 tokens total).

### 37 · `getrows-narrow-batched` — `CUDA`

`get_rows` puts one row in one block and splits the columns across the block's
threads, so a 64-element row leaves 192 of 256 threads idle. The narrow-row
kernel patch 14 added fixes that, but only fired for
`ne00 <= 2 && ne11 == 1 && ne12 == 1`, which excludes every batched gather —
including the per-query chunks of a sparse-attention indexer (`ne11 = 64`).

Generalized to batched gathers (the index arithmetic is the generic kernel's)
and routed to for any row width at or below `GGML_CUDA_GETROWS_FLAT_MAX`
(unset = off; the existing condition stays). It only copies elements, so output
is **bit-identical** (verified: max |dlogprob| = 0 over 80 top-5 samples).

4,097-token append at 40k depth: 25.8 → **23.1 s (−10.4%)**. Fresh 40k prefill
over three interleaved rounds: 231.7 → **210.0 s (−9.4%)**, distributions fully
separated. Decode unchanged.

### 38 · `getrows-q4-0-block` — `CUDA`

The quantized `k_get_rows` has each thread read a 2-byte scale and one nibble
and write two halves 16 elements apart; a block writes 1 KB. Gathering 131k
rows × 512 elements from a q4_0 KV cache that way ran at **89 GB/s**, a fifth of
what the card does in practice.

Adds a q4_0 → F16 kernel with one thread per q4_0 block: 32 elements
dequantized and written as four 16-byte stores of 64 contiguous bytes, so a
warp's writes coalesce completely. The dequantization is `dequantize_q4_0`'s own
expression, so output is **bit-identical**; unaligned destinations fall back, and
the kernel exists only for an F16 destination, which is what patch 45's gather
asks for. `GGML_CUDA_GETROWS_Q4_0_BLK=1`.

Fresh 40k prefill, on top of patch 37: 208.3 → **187.3 s (−10.1%)**, three
interleaved rounds, distributions separated. Decode unchanged (it gathers the
same rows but far fewer of them).

Together 37 and 38 take a fresh 40k prefill 231.7 → **187.3 s (−19.2%)**.
GET_ROWS was 26.6% of prefill GPU time not because of arithmetic but because
the launch geometry did not match the work.

### 41 · `top-k-radix-select` — `CUDA`

Patch 28 gives `top_k` a partial path for small k (k <= 16); above that it still
falls back to CUB's full segmented sort. A sparse-attention indexer selects
k = 2,051 of 40,544 scores, so it sorted all 40,544 to keep the first 2,051:
**133 µs to read 162 KB** (161 µs at 210k depth, 65.6 ms for a 4,096-row prefill
chunk).

Adds a radix select for large k (`GGML_CUDA_TOP_K_SELECT=1`), in the same block
patch 28 guards off for HIP and for builds whose CUB provides `DeviceTopK`, and
only for rows of at least `CUDA_TOP_K_MIN_NCOLS` = 4,096 columns:

- the comparison key is a 64-bit packed `(ordered_key << 32) | ~idx`, so
  unsigned descending order *is* "key descending, index ascending" and matches
  the stable sort. Packed values are unique within a row, so the k-th element is
  a single element and **no tie-breaking is left**
- 8-bit digits from the top; the bytes of `~idx` above the index's top bit are
  constant, so those rounds are settled up front (6 rounds at 40,544)
- bin scan is a 256-bin parallel suffix sum. A first version had thread 0 walk
  the bins serially, which raced on `states[row].rank` at large `nrows`; one
  barrier fixed it
- the output is **index-ascending**: chunks map to contiguous ranges and are
  packed with a warp ballot over a scanned base, so the result does not depend
  on block execution order

Index order is not cosmetic. An unordered (atomicAdd) output changed model
output — max |dlogprob| 5.22 at 40k — because the sparse gather of patch 45
collects K/V in top-k order, so flash attention's accumulation order follows it,
and atomic order is not reproducible run to run. With ascending order: the
selected set **matches the full sort on every shape**, including synthetic rows
with 4 and 64 forced ties; two runs of the same server are **bit-identical**
(max |dlogprob| 0 at 40k and 210k); and a 26-question × 3-depth long-context
benchmark matches question for question (p = 1.000).

Kernel 133 → **74.8 µs** at the decode shape, 65.6 → **28.6 ms** at 4,096 rows.
End to end: decode 40k 48.6 → **47.5 ms (−2.3%)**, prefill of 2,050 tokens
126.8 → **131.1 t/s (+3.4%)**, decode at 210k unchanged.

### 48 · `fuse-rms-norm-scale` — `CUDA`

Since `v0.5.0` gated delta-net normalizes q and k as `scale(rms_norm(x, eps/n),
1/sqrt(n))` instead of `l2_norm` (ggml-org/llama.cpp#28068), so 17's and 18's L2-norm halves stop
firing on these models and each layer launches four small kernels. It matches any
run of contiguous F32 RMS_NORM → SCALE pairs, not only those, and runs it as one
launch of a kernel built from `rms_norm_f32<256>`'s implementation, with the
scale applied in its epilogue the way `scale_f32` applies it, so the output is
bit-identical (9B, 256 greedy tokens: same tokens and same top-5
log-probabilities with and without it). It does not fuse when an output would
alias any input, when the RMS_NORM result has another consumer, or for rows of
1,024 or more, nor while the graph runs concurrent streams
(`GGML_CUDA_GRAPH_OPT=1`). `GGML_CUDA_DISABLE_FUSE_RMS_SCALE=1` turns it off.

`llama-bench` tg64, four rounds, order reversed every other round: 9B 72.93 →
73.72 t/s (+1.1%), 35B-A3B 79.39 → 81.34 (+2.5%).

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

### 39 · `gdn-lanes-per-column` — `CUDA`

The gated delta-net kernel spreads one state column over a whole warp, so every
token pays a full warp reduction per column. Making lanes-per-column a template
parameter and using 16 lanes halves the reductions per token and raises the
FMA-to-shuffle ratio — but it also halves the parallelism, which only pays once
there are enough tokens to hide it. Gated on the token count:
`GGML_GDN_LPC=8|16` selects the lane count (unset = a full warp as before) and
`GGML_GDN_LPC_MIN_TOK` the threshold (default 32). Only `S_v == 128` has the
narrow instantiations, so on any other head size the variable is ignored.

Re-measured on the published build by flipping only this variable, two rounds
with the order reversed and both arms cooled to ≤56 °C: a fresh 40k prefill
**165.0 vs 172.5 s (−4.3%)**. During development it measured −4.9% (161 / 167 →
154 / 158 s).

Decode is untouched by construction: the override needs `n_tokens >= 32`, and a
decode step without speculation is one token. The same two rounds did show decode
at 40k moving 49.2 → 47.7 ms/token, which is therefore this protocol's
server-to-server spread and not the flag — worth knowing when reading the decode
figures of the other two variables re-measured the same way (45 and 46), where the
flag *can* act but the spread is the same size.
Output is not bit-identical — the reduction order changes — and a paired
per-chunk perplexity comparison against the unchanged build gives t = +0.25, i.e.
no detectable difference.

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

Merged with upstream's two-arena retention of the previous graph (`gf_res_prev`
is now a `std::array`, and `gf_res_prev_active` marks the arena holding the main
scheduler's current allocation). Decode through a slot does not rewrite
`gf_res_prev_active`.

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
TOPK_MOE_ROWS_PER_BLOCK`, = 8 upstream on `v0.4.0`) means a verify batch of
`n_draft + 1 = 5` still falls inside it — the exception's own edge is at
`n_tokens = 9`, not 5 — so it does not explain the measured boundary: both a
dense and an MoE model stayed bit-identical at `n_tokens <= 4` and changed
output at `n_tokens = 5`. The `n_tokens <= 4` cap is that measurement, not
something read off the topk-moe exception.

4 is the default because the asymmetry is severe: running out of VRAM means the
model does not load, while 0.38 points — real, and well above the ±0.07%
resolution established in [benchmarking.md](benchmarking.md), but small — is not
worth that risk. Raise it with `LLAMA_DEC_SLOTS=8` where there is headroom; `=0`
restores upstream behaviour.

`LLAMA_DEC_MAX_TOK` defaults to 4, and that number is load-bearing:
`ggml_cuda_check_fusion_memory_ranges` decides fusion from actual tensor
addresses, so a different buffer layout changes which fusions fire and
therefore changes the output. topk-moe skips that check at
`nrows <= TOPK_MOE_ROWS_PER_BLOCK` (= 8 on `v0.4.0`), but the default of 4 is
not read off that bound — it is measured bit-identical at `n_tokens <= 4`,
changing at 5 and above.

---

### 32 · `sched-split-prefetch` — `host`

When a graph alternates GPU and CPU splits, the GPU sits idle through every CPU
split even though part of the *preceding* GPU split produces nothing that split
needs. This cuts the GPU split at the last node the next CPU split consumes,
queues that head, starts the async copies of the CPU split's inputs, and queues
the tail behind them, so the tail overlaps the CPU work. The copies are waited
for with their own event, not by synchronizing the backend. The backend now sees
two graph submissions where it saw one, so a backend that caches a captured
graph per context re-captures on every evaluation; the measurement below is net
of that.

Off by default (`LLAMA_SCHED_PREFETCH=1`). On a 48-layer MoE with 48 expert
slots resident on the GPU and the remaining experts computed on the CPU:

| | off | on |
|---|---:|---:|
| decode at 40k | 66.4 ms | **59.4 ms (−10.5%)** |
| decode at 210k | 62.0 ms | **56.1 ms (−9.5%)** |

What is overlapped is the tail of the GPU split — the cached experts' matmuls.
An earlier revision measured **no effect at all** with 34 resident slots,
because there was nothing large enough to hide; that is why this ships off and
why a per-model measurement is the only way to know.

### 33 · `sched-weight-prefetch` — `host`

Prefill of an MoE model whose experts live in host memory spends its time in
three back-to-back host-to-device transfers per layer with almost no compute
between them (the three GPU splits of a layer are 1, 2 and 150–230 nodes). This
uploads the *next* splits' offloaded expert weights on a separate backend and
stream while the current split computes, then moves each into its input
allocation with a device-to-device copy.

The staging memory can be borrowed from a buffer that belongs to something else
through a new API, `ggml_backend_sched_set_staging_area(buffer, size)`: the
scheduler uses it only in graphs that do not reference it, and publishes an
epoch counter and a high-water extent so the owner can see which bytes were
clobbered and repair them. The uploads run on their own backend, which no
graph-level synchronize covers, so the owner waits for them with
`ggml_backend_sched_staging_synchronize()` before it repairs anything, and drops
the registration with `ggml_backend_sched_clear_staging_area()` before the buffer
goes away. The producer table holds 16 backends; a scheduler that does not fit
leaves weight prefetch off rather than borrow memory nothing would wait for.
Staging is confined to the device that owns the buffer: a split on
another device would have to take the weight through a host copy, which does not
honour the upload event. In the measured configuration the ~1 GB comes from an
expert cache that prefill does not use.

Off by default (`LLAMA_SCHED_WPF=<MiB>`; 1000 in the measured configuration).
Appending to a 40k context, and a fresh 20k prompt:

| | off | on |
|---|---:|---:|
| 1,025 tokens | 11.9 s | 11.8–12.0 (±0, the refill cancels the gain) |
| 2,049 tokens | 17.4 s | **16.6 s (−4.6%)** |
| 4,097 tokens | 29.0 s | **27.1 s (−6.6%)** |
| fresh 20k | 123.4 s | **115.7 s (−6.2%)** |

VRAM does not grow. The peak while filling a 262k context *fell*, 15,231 →
14,009 MiB: the prefetch replaces a partial-copy path that synchronized the
device to read `ids` on every split.

### 34 · `mul-mat-id-negative-ids` — `host`

`mul_mat_id` assumes every id selects a real expert row. Two changes let a
negative id mean "this row contributes nothing": the CPU backend writes zeros
for it instead of indexing out of bounds, and the scheduler's used-expert bitset
skips it (and tolerates a split where no expert is used at all).

No effect on its own — it is what lets patch 46 route the experts it did not
cache to the CPU by masking them out of the GPU call with −1, and what keeps a
graph where every expert of a layer missed the cache from asserting.

### 47 · `sched-split-inputs-cap` — `host`

Caps the number of inputs a split takes, which upstream stopped doing in
`v0.5.0` (ggml-org/llama.cpp#28387): a node whose new inputs would take the split past
`GGML_SCHED_MAX_SPLIT_INPUTS` starts a new one. Without it a split takes
any number of inputs, and on a graph whose GPU nodes read large host-side
inputs (the KQ mask and the indexer scores of a sparse-attention model) the
device copies of all of them stay live across one long split. On the
offloaded sparse-attention MoE at ctx 262,144 / ubatch 6144 the CUDA0 compute
buffer went from 2,596 MiB to 9,512 MiB and the context no longer fit on a
16 GiB card; with this patch it is 2,600 MiB. The cap is fixed at
`GGML_SCHED_MAX_SPLIT_INPUTS` and counts each distinct new input a node brings, so it
does not drift with the input list's capacity. It is **off by default** and
enabled with `GGML_SCHED_SPLIT_INPUTS_CAP=1`: it moves split boundaries on any
graph that reaches the cap, and only this workload has been shown to need it.

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

Patches 42 and 44–46 target one architecture: a hybrid gated-delta-net /
sparse-attention MoE (`qwen4exp`, i.e. Qwen3.8-Flash-Next) — 48 layers, 512
experts with 10 used, a gated delta-net block in most layers and a quantized
sparse-attention (QSA) path in the rest. The numbers below come from
UD-IQ3_XXS weights, a q4_0 KV cache and a 262,144-token context on one P100,
with the experts in host memory. Unlike the rest of the set they change what the
graph computes, not only how a kernel runs, so each one names what it does to
the output.

The tree these numbers were taken on also carried one change that is **not** in
this set: `VDR_Q6_K_Q8_1_MMVQ` raised from 1 to 2, which changes Q6_K matvec
numerics and was never measured on its own, so it was left out. It is not free of
consequence for the figures — this model keeps 94 tensors in Q6_K — so it was
measured afterwards, against the published set, in the production configuration,
two rounds each: a fresh 40k prefill 238.0 vs 240.2 t/s (the dropped line ahead by
0.9%, on a spread of 1.6% against 6.7% within the arms), decode at 40k best of
five 49.6 vs 51.2 ms/token and one 31-token ubatch 24.1 vs 24.7 ms/token (the
published set ahead on both). Nothing separates them beyond this protocol's
spread, and the metrics disagree about the sign. Model output does change: with that one line restored, this
set reproduces the measured tree **bit for bit** (16/16 positions, max
|dlogprob| 0), which is also how the fixes made during review were shown not to
alter output.

**What 32–46 were worth on this model** (measured on `v0.4.0`, where they were
fifteen patches including the since-deleted 43; on `v0.5.0` the same configuration also needs
`LLAMA_QWEN4EXP_HC_FOLD=1` and `GGML_SCHED_SPLIT_INPUTS_CAP=1`), both arms built
from this repository and measured after the split, two rounds with the arm order reversed:

| | stock `v0.4.0` | 01–31, `--n-cpu-moe 44`, ubatch 1024 | 01–46, `--n-cpu-moe 48`, ubatch 6144, the variables below |
|---|---:|---:|---:|
| fresh 40k prefill | 413 s (97.6 t/s) | 431 s (93.5 t/s) | **177 s (227.4 t/s)** |
| decode at 40k, best of five | 115.9 ms/token | 101.4 ms/token | **48.9 ms/token** |
| VRAM peak while doing it | 14,559 MiB | 14,635 MiB | 15,429 MiB |

Against stock that is **+137% on decode and +133% on prefill**; against the
published 31-patch set, −58.8% on prefill and −52% on decode. Stock does start at
this context length on this card, at the same options the 31-patch set runs.

The VRAM row is the peak sampled every two seconds through that 40k run, which is
not the worst case. The worst case for the right-hand configuration is a 262k
context restored and then decoded speculatively; that was measured during
development at **15,393 MiB with no failed allocation**, and it is the number to
budget against.

Protocol: the two patched columns are two rounds with the arm order reversed. The
stock column is three consecutive runs rather than interleaved with them (prefill
423 / 413 / 403 s, a spread that is the host page cache warming to the 41 GB of
weights), its first two rounds are best of five decode reps and the third is best
of three, and its VRAM figure comes from that third run, the only stock round
that sampled memory.

The arms differ in server options as well as in patches, deliberately: one more
offloaded expert layer, a six-times wider ubatch and 60 resident expert slots are
what the code makes reachable, and the left column is the configuration the
published set can actually run. The extra 794 MiB is the cache; it stays under
the 15,697 MiB peak this context length was already validated at.

Long-context accuracy over the same span went 59/78 → 62/78 (26 questions at
three depths, McNemar p = 1.000 on every comparison), measured during
development rather than on this build.

### 42 · `qwen4exp-hc-exact` — `model`

Hyper-connections are about 23% of decode GPU time on this model: four low-rank
matmuls per layer plus the element-wise glue. Two changes remain:

- fold the `1/hc = 0.25` SCALE into the weights on the first graph build — the Q8_0 `down`
  block scales and the F32 `inject` values. A tensor is folded only if it is on a
  device and every scale survives the division exactly and stays an F16 normal.
  That makes the decode path (MMVQ, and F32 for `inject`) **bit-identical**; a
  prefill matmul that runs in cuBLAS with F16 accumulation can still round its
  partial sums differently at a quarter of the magnitude. Host-resident weights
  are skipped whatever their scales, so the 122 of 193 that qualified is not a count
  of rounding failures. The rest keep the original path, per tensor
- drop the `CONT` before the stream average. Since `v0.5.0` only the final head
  mix (`il = -1`) reaches that code: the layers take upstream's fused
  hyper-connection op (ggml-org/llama.cpp#28901). **Bit-identical**

The gamma reshape this patch used to make is upstream since `v0.5.0`, which reads
gamma as `[n_embd, hc]` so RMS_NORM and MUL are adjacent (ggml-org/llama.cpp#28896).

The original three-change patch was worth about **−1% decode** (0.5 ms); what
remains has not been re-measured on its own. Event-level profiles had made
the glue look larger than it is; per-op host synchronization inflates small
kernels.

The fold is **off by default** and enabled with `LLAMA_QWEN4EXP_HC_FOLD=1`,
because it rewrites the weights in place. `build_lora_mm` adds an adapter's
contribution after the base matmul, so an adapter on a folded `hc_*` tensor would
no longer be divided by `hc` and land four times too strong: building a graph
with such an adapter at a non-zero scale aborts. Saving the model after the fold
is refused (logged, nothing written) rather than write the divided weights.
Fine-tuning through `llama_opt` builds graphs too, so with the fold on the
`hc_*` weights would be optimized at a quarter of their scale; leave it off
for training. The `CONT` removal
is on with no kill switch; its bit-identity rests on the whole-set comparison at
the top of this section rather than on a check of its own.

### 44 · `qwen4exp-qsa-block-key-cache` — `model`

The sparse-attention indexer recomputes pooled, normalized and rotated block
keys for the entire context on every ubatch, which makes decode cost grow with
depth. This keeps them per layer as F32 `[128, n_kv/4]` — 384 MiB at ctx 262,144
— and refreshes only the blocks whose cells changed. Invalidation hooks sit on
`clear`, `seq_rm`, `seq_cp`, `seq_keep`, `seq_add`, `seq_div` and `state_read`.
**Almost all of decode's depth dependence disappears.** `LLAMA_QSA_CACHE=0`
disables it.

Known limitation: the incremental refresh only writes *complete* blocks, so
while the newest positions form a partial block — up to `ratio - 1` of them —
that block is scored against whatever its cache row already held (zeros on a
fresh context, the key of a longer previous context otherwise). The full-refresh
path, which runs whenever the cache falls further behind than one ubatch, writes
every block including the partial one. The accuracy figures on this page were
taken with this behaviour in place.

A block is only fixed once its cells are known: the cache is used for a single
sequence whose cells hold positions 0..n−1 without gaps, so block *b* is
position block *b*. Two traps worth recording: `ubatch.is_pos_2d()` is true for
text input on an mrope model, so using it as the enable condition silently kept
the cache from ever being used; and on the cached path the per-cell index
tensors are never allocated (`data == null`), which faults in `set_input` unless
the setter writes to scratch instead.

### 45 · `qwen4exp-qsa-sparse-gather` — `model`

Upstream ships this architecture with sparse attention stubbed out — `TODO:
enable sparse attention when we are ready`, the graph is dense flash attention
with a top-k mask —
and the upstream sparse path needs Turing mma, so Pascal with a q4_0 KV cache
cannot take it. This gathers the selected cells' K, V and mask with `get_rows`
and feeds flash attention one query per stream slot. The indexer scores and the
top-k are chunked per query row (`LLAMA_QSA_CHUNK`, default 32), which removes
the `[n_kv × n_ubatch]` intermediate: **compute buffer 3,773 → 1,063 MiB** at
chunk 32. `LLAMA_QSA_GATHER=0` returns to the dense path; so does a build or context
without flash attention, more than one stream, a 4-D mask, or a KV cache shorter
than `2*(top_k + ratio - 1)`, all of which the sparse path requires.

This patch is **on by default and changes output**: gathering the selected cells
and running attention over them accumulates in top-k order, where the dense path
accumulated over the whole KV behind an additive mask. The selection is the same —
patch 41's entry depends on that order being deterministic — but the sums are not
bit-identical to upstream.

The gather writes F16 directly: with an F16 destination, CUDA's `get_rows` does
q4_0 → float → half in one pass instead of an F32 write plus a CPY. The rounding
is the same, so this part is **bit-identical**, and prefill gains 7% at 20k and
3.4% at 40k while decode loses about 1%. Only the K/V gathers take it: the mask
gather has no buffer to inspect when the graph is built, so it keeps the F32 write
and an explicit cast, which is also what any CPU placement needs. (Gathering the raw q4_0 bytes instead was tried and abandoned: the
compute buffer grew with the chunk count, 17.8 GB at ubatch 4096, and the server
would not start.)

`LLAMA_QSA_PAD=1` pads the gathered K/V and mask to a multiple of
`FATTN_KQ_STRIDE` so flash attention takes the vector kernel instead of the tile
kernel. Re-measured on this build by flipping only that variable, two rounds with
the order reversed: a fresh 40k prefill **165.5 vs 178.0 s (−7.0%)** and decode
at 40k **47.4 vs 49.0 ms/token**, During development it measured −7.8% on prefill and −18% on 262k
speculative decode, at a VRAM peak that did not move; the re-measurement did not
sample VRAM. Its two rounds were also not temperature-matched (the padded arm
started a round at 48 °C against 56 °C for the unpadded one), and the decode
figure is the same size as the spread discussed under patch 39, so read the
prefill number and treat the decode one as noise. Output changes, and it changes for the better: the vector kernel
accumulates VKQ in float2 where the tile kernel uses half2 on NVIDIA, so
relative RMS against a CPU F32 reference is *lower* than the unpadded path in
every shape measured, and a paired per-chunk perplexity comparison gives
t = −0.36.

`LLAMA_QSA_TREE_CONCAT=1` builds the chunked output with a pairwise concat tree
instead of a left-leaning chain.

### 46 · `qwen4exp-moe-expert-cache` — `model`

With 48 layers of 512 experts the routed weights do not fit on a 16 GiB card, so
`--n-cpu-moe 48` computes them on the CPU. This keeps the recently used ones on
the GPU instead: C slots per layer, filled from a flat count over a ring window
of recent tokens and replanned every so often. The routing history is written
out to a device tensor and collected at the next `set_input`. Experts that hit
the cache are computed by a GPU `mul_mat_id`, the misses by the CPU one with −1
in the id (patch 34), and the two are added. The cache refuses to start on a
model whose experts need anything the cached path does not reproduce: fused
gate/up tensors, a non-zero SwiGLU clamp, per-expert scale tensors, or experts
that are not all host-side.

`LLAMA_MOE_CACHE=<slots>` enables it (off when unset), with
`LLAMA_MOE_CACHE_INTERVAL` (default 32) the replan period,
`LLAMA_MOE_CACHE_WINDOW` (default 1024) the ring window,
`LLAMA_MOE_CACHE_MAXTOK` (default 8) the ubatch width it stays active for — it
is also capped at `GGML_OP_OFFLOAD_MIN_BATCH − 1`, since above that threshold
the CPU experts are offloaded to the GPU anyway, and at `LLAMA_MMVQ_MMID_MAX`
(patch 35), because the shared zero slot repeats ids within a token and only
MMVQ tolerates that (4 when that variable is unset, see below). The cache is not
built at all if the caps leave nothing, or if the model routes experts in groups,
which `build_moe_cached` does not reproduce. `LLAMA_MOE_WARM`
(default 32) the number of tokens after a prefill during which the replan period
is shortened to 8.

Slot count, decode ms/token, interleaved two rounds:

| slots | 40k | 210k |
|---:|---:|---:|
| 48 | 55.2 ms | 53.0 ms |
| 54 | 51.9 | 53.1 |
| 60 | **51.0** | **52.4** |

A slot costs about 92.5 MiB across the 48 layers. A longer replan period is
better than the default. Re-measured on this build with the slot restored so the
arms see the same context, two rounds with the order reversed: 128 gives
**47.45 vs 48.65 ms/token (−2.5%)** against 32, and one of the two rounds showed
no difference (47.2 / 47.7 against 49.8 / 47.5). During development it measured
−3.8% (44.3 → 42.6 ms at 40k) with the 26-question long-context bench answering
identically.

The cache is cold right after a fresh prompt, and prefill routing does not
predict generation routing well enough to warm it (that was tried and dropped).
Shortening the replan period for the first 32 generated tokens instead takes the
first window's hit rate from 7.2% to **46.6%**: 94.2 → **83.7 ms/token** for a
32-token generation (−11%) and 75.8 → **70.0** for 64 tokens (−8%), with the
steady state unchanged. Each of those points needs its own server: measuring
several generation lengths against one server lets the earlier generation warm
the cache and reverses the result.

Two graph-order changes ride along, both bit-identical: the shared expert and
the hyper-connection inject matmul are expanded *before* the CPU split rather
than after it, so they run under patch 32's overlap (−0.5% at 40k, −1.0% at
210k; `LLAMA_INJECT_LATE` restores the old order for the inject).

The cache also lends its buffer to patch 33 as a staging area during prefill,
and repairs the slots it loses: on an epoch change it invalidates the layers
inside the clobbered extent, re-zeroes the shared zero slot, re-sends the maps
and refills on the next decode.

The clamp against the MMVQ ceiling uses `LLAMA_MMVQ_MMID_MAX`, which is the
override patch 35 applies to every type. The built-in ceiling is per type and
per architecture, and outside sm_60 it is below 8 for several types (Turing and
later: 5 for Q3_K, 6 for IQ3_S, 7 for Q2_K, IQ2_S, IQ3_XXS and MXFP4; the AMD
tables are lower still). The host side cannot read that table — it reads the
variable, capped at 32, and cannot see the device's own `1024/warp_size` limit,
which is 16 on a 64-lane device and is where the two limits diverge. Unset, the
cache therefore assumes the smallest entry in any table (4). To get a wider
batch on any architecture other than the one this was measured on, **set
`LLAMA_MMVQ_MMID_MAX` to a width that device and those expert types take**: set
too wide, a ubatch past the real ceiling sends the GPU expert matmul to MMQ,
which folds together the repeated ids the shared zero slot produces and leaves
the rows of the folded-away experts unwritten.

Accuracy over the whole configuration is 62/78 on the 26-question × 3-depth
long-context benchmark, against 59/78 for the same model without any of this. The
two ends are not one experiment: 59/78 is the pre-patch production build, 62/78
was taken during development on a 56-slot configuration, and the p ≥ 0.5 figure
comes from the 59/78-against-61/78 comparison in between. Nothing here was
re-scored on the published build.
