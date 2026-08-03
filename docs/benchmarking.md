# How these were measured

日本語版: [benchmarking.ja.md](benchmarking.ja.md)

Most of the patches here are worth between 0.5% and 2%. Measuring that reliably
is harder than writing them, and several were nearly rejected — or nearly
accepted — because of a broken measurement. This is what it took.

## Measure in the regime you ship

Three separate wrong conclusions came from benchmarking under conditions that
differed from the one the patches actually run in.

**Thermals.** A P100's boost clock sags from 1328 to 1101 MHz (−17%) over a long
run. Measuring candidates sequentially therefore favours whichever ran first, by
more than the effect being measured. Fix: interleave the baseline between
candidates, and wait for the GPU to drop below a fixed temperature before every
run. With that, the baseline reproduces to ±0.02% on prefill.

**Greedy sampling.** Any change that moves the speculative acceptance rate must
be measured with a *realistic* sampler. One patch looked like +7.7% under
temp 0 / top-k 1, because greedy pushes acceptance to 0.99 and unfairly favours
deeper drafts. With a realistic sampler it was +2.2%.

**The wrong harness.** `llama-bench` has no speculative decoding. One change
measured +1.4% there and **+0.01%** in the real workload, because without a draft model
every expert matmul is `ncols_dst = 1`, which is precisely the case the change
addressed; in the real workload that traffic moves to a different kernel.

## Know your noise floor before trusting a number

"Same-binary A/B varies by ±0.4%, so anything under 0.5% is unmeasurable" was
wrong twice over. Running a **null A/B** — the same binary as both arms — gave:

| | SD |
|---|---:|
| geometric mean, 3 prompts | 0.169% |
| prompt A | 0.113% |
| prompt B | 0.151% |
| **prompt C** | **0.555%** |

Two errors. First, ±0.4% was a *maximum*, not the resolution of a mean; over
four rounds the mean resolves to ±0.165%. Second, **one prompt was producing
about 90% of the variance** (0.555² of 0.343 total) — and not by varying
constantly, but bimodally: 2 of 16 runs came in ~1% slow.

Dropping that prompt took the SD from 0.169% to 0.068% — ±0.067% over four
rounds. That is **2.5× better than the measured three-prompt figure, and 6×
better than the ±0.4% I had been budgeting for**, for free. Clocks were innocent (all 74 samples
at 1328 MHz, throttle reasons 0x0 throughout).

Validated against a known effect: a kill-switch A/B measured +0.723% ± 0.041%
over 8 rounds, and +0.720% over the first 4. A by-product was that patch 20
(`fuse-add-unary-mul`) was really worth +0.72%, not the +0.47% first reported.

## Prefer one binary with a runtime switch

Where possible each patch has a kill switch (`GGML_CUDA_DISABLE_*`,
`LLAMA_*=0`), so both arms of an A/B are the **same binary**. That removes code
layout, allocator state, and build nondeterminism from the comparison at once.

When two binaries are unavoidable, build the control by removing only the patch
from the same tree — then check the direction that must actually hold:
**re-applying it has to reproduce the reference build byte for byte.** If it does
not, your control differs from that build by something other than the
patch, and the difference you are measuring is not the one you think it is.

**Always include an arm that should be unchanged.** One round of results had to
be discarded because the scratch tree still carried a previously rejected
change; the tell was that the "no-op" arm came in 3.4% below the reference build
with a different register count. A control arm that fails to reproduce the
baseline is the cheapest bug detector available — do not proceed without it.

## Reading profiles

**Split by grid X, not by kernel name.** Per-call cost inside one kernel name
can vary 27× between tensors. Grid X is rows / rows-per-block, which usually identifies
the tensor uniquely. Aggregating by name once hid the entire finding behind an
average.

**Exclude prefill from a decode profile.** Selecting generation regions by
"≥200 kernels and >50% busy" also catches prompt processing (30% of the trace,
dominated by dequantize + cuBLAS hgemm) and the startup memset. Everything gets
diluted by about a third.

**Kernel launch counts identify pairs.** Two kernels firing exactly the same
number of times is strong evidence they are one-to-one. That is how the
duplicate q8_1 quantization and several fusion candidates were found.

## Instruction counts, occupancy, bandwidth

**Instruction count is only meaningful at constant registers.** One rewrite cut
a kernel by 20.7% of its instructions and ran 38% *slower*: register use hit the
cap, and the recomputation needed to stay under it cost more than the
instructions saved.

More generally, on the sm_60 Q4_1 HFMA2 GEMV (patch 12's "a16" kernel) every
instruction reduction measured neutral or negative — including one that removed 63% of the loop's instructions while
moving zero extra bytes, for −6.4%. The arithmetic was already hidden under
memory latency, and was itself acting as the cover.

**Check the register limit before predicting occupancy.** A 32-thread block does
not give 32 blocks per SM if the kernel uses 94–140 registers per thread: a P100
SM holds about 640 such threads, not 2048. Predicting 4.2× from thread and block
caps alone produced 1.75×.

**Occupancy changes confound register changes.** Raising occupancy from 25% to
31% via `__launch_bounds__` measured −4.8%. Forcing the resulting 96-register
code back to the *original* block count did not recover it: the loss was the
+30 instructions of address rematerialization the register cap forced, not the
occupancy.

**Derive bandwidth, do not assume it.** Divide the weight bytes actually read per
forward pass by the measured time and compare against a *measured* ceiling
(606 GB/s on a P100 — 83% of the 732 GB/s theoretical figure).

## Isolated microbenchmarks did not predict this kernel

Twice, a standalone benchmark that reproduced the real kernel's timing at the
real tensor shape still mispredicted the real change by ~10 points:

| Change | Microbenchmark | Real |
|---|---:|---:|
| uniform rows-per-block 4 → 8 | +30–40% | −3.0% |
| the same, only for large row counts | +6.8% | −0.9% |

The second attempt included the real arithmetic and was calibrated at 4,096
rows. It still missed. The cause was never isolated (candidates: L2 state left
by the preceding kernel, different K, cache interference from other kernels).

Microbenchmarks were reliable for *ranking variants of one kernel in isolation*
and unreliable for *predicting end-to-end change*. Treat them as a filter, not
as evidence.

## Fusion has its own traps

**Compare the producer's and consumer's grids before fusing.** Folding
`quantize_q8_1` (grid 16) into `rms_norm_f32<1024>` (grid 1) runs the same work
at one sixteenth the parallelism: no work was added and it still grew by
1.90 µs. Measured −1.12%, rejected.

**Adding a template parameter slows down callers that do not use it.** The same
attempt cost +0.26 µs across all 14,102 calls of an unrelated path.

**Indexing a kernel parameter struct at runtime spills it to local memory.**
`p.cx[blockIdx.y]` produced a 128 B stack frame; at 24,576 threads that is 3 MiB
of extra traffic and the "fused" version ran 2.6× slower than the unfused pair
(38.7 µs against 14.7). `#pragma unroll` to
constant indices only: 0 B, and 14.7 → 8.2 µs. Taking `blockIdx.y` into an `int`
local before comparing is part of the fix.

**`ggml_view_*` nodes sit in the graph.** Kernels that look adjacent in a trace
can have VIEW nodes between them, so collecting "consecutive nodes with the same
op" fires *never*. Skip no-ops explicitly. Confirm firing from kernel names in a
profile — `GGML_LOG_INFO` from the CUDA backend does not appear under
`llama-batched-bench`.

**Deferred reads move when the data is read, not just when it is computed.** If
anything writes the source in between, the fused version reads the newer value.
This is not hypothetical: `build_rs` assembles writes into the same state buffer.

**Buffer layout changes fusion decisions.** `ggml_cuda_check_fusion_memory_ranges`
tests actual addresses, so a change that only moves allocations around can
change which fusions fire and therefore change the output. Both results are
correct, but they are not bit-identical, and a hash-based regression check will
flag it.

## One clean run is not evidence

Three habits, all cheap, each of which caught a real bug in this set.

**Run the whole suite, not the failing case.** A `test-backend-ops` failure in
SOFT_MAX did not reproduce under `-o SOFT_MAX`; it appeared only in a full run,
because the damage was done by state an earlier op left behind. Narrowing to
the failing case is the natural debugging move, and it makes this class of bug
invisible.

**Run it more than once, and read *where* it fails.** Consecutive full runs
failed at different ops with very different error magnitudes — 2e-4 in
SOFT_MAX, 0.54 in MUL_MAT_ID. A precision problem is stable and stays in one
op. A failure that moves, with an error far too large for rounding, is memory
being reused, not arithmetic.

**A crash can hide a wrong answer.** An abort partway through the suite left
the rest unrun; removing it exposed a silent incorrect-result bug that had been
sitting behind it the whole time. Reaching "no crash" is not reaching
"correct" — the run has to complete, repeatedly, before a pass means anything.

## Do not extrapolate hit counts linearly

Graph reuse looked like a large win by extrapolation: 341 hits per 1000 were
worth +1.92%, so the ~540 additional hits from raising reuse to 88% should have
been worth about as much again.

It was worth +0.21%. A *true* hit skips build, reset, split and allocation; the
partial hit that the extra slots produced skips only build. The two differ in
value by **15×**. Hits are not fungible — check what each one actually skips.
