# llama-cpp-p100-patches

44 performance patches for [llama.cpp](https://github.com/ggml-org/llama.cpp),
developed and measured on a **Tesla P100 (GP100, sm_60)**.

日本語版: [README.ja.md](README.ja.md)

```console
$ nix build github:shinbunbun/llama-cpp-p100-patches#llama-cpp-sm60
```

GP100 is the reason this exists. Alone among Pascal chips it has no DP4A and
has full-rate HFMA2 — so llama.cpp's quantized matmul paths fall back to
emulation and cuBLAS, while the one instruction the card is genuinely good at
goes unused. The patches named `sm60` exploit that asymmetry.

**But only 8 of the 44 are gated to Pascal-era hardware.** One more changes an
unconditional constant that every GPU sees. The remaining 35 are not
hardware-scoped at all — kernel fusions, index-arithmetic fixes, two host-side
sampler paths, four scheduler patches, two `top_k` paths that avoid sorting a
whole row, gather kernels whose launch geometry matches the work, and two lookup
tables staged in shared memory — though six of those only fire on gated
delta-net models and five need a specific model architecture.

Patches 32–46 come from one workload: a 48-layer sparse-attention MoE with its
routed experts in host memory, at a 262,144-token context. Most of them are
**off by default** behind an environment variable, because what they are worth
depends on how much of the model is offloaded. On that model, and with both arms
built from this repository, they are worth **−58.8% on a fresh 40k prefill**
(431 → 177 s) and **−52% on decode at 40k** (about 101 → 49 ms/token) against the
31-patch set — the arms differ in server options too, which is the point of them;
see [docs/patches.md](docs/patches.md).
Every row in the table below carries a scope tag, and
[docs/patches.md](docs/patches.md) is grouped by it.

Every patch has a measurement behind it, and the measurements — including the
alternatives that were tried and rejected — are in the comments the patches add
to the source, except for 32–46, whose measurements are in
[docs/patches.md](docs/patches.md) instead.

## What the set is worth

Against stock `b10133` on the same card, same models, same server options:

| Model | Stock | Patched | |
|---|---:|---:|---:|
| Qwen3.5 9B dense (Q4_1, MTP head Q4_1) | 76.99 t/s | **136.11 t/s** | **+76.8%** |
| Qwen3.6 35B-A3B MoE (Q2_K_XL, dense layers Q5_1) | 67.01 t/s | **120.57 t/s** | **+79.9%** |
| Qwen3.8-Flash-Next MoE (UD-IQ3_XXS, q4_0 KV, ctx 262,144, 40k deep) | 8.63 t/s | **20.47 t/s** | **+137%** |

End-to-end decode throughput from `llama-server`, geometric mean over three
prompts, MTP speculative decoding (n-max 4, p-min 0.75) and a realistic sampler
(temperature 0.7, top-p 0.8, top-k 20). Arms interleaved, order reversed for the
second half, GPU cooled to ≤58 °C before each arm: six rounds for the dense model
(SD 0.19% stock / 0.08% patched), four for the MoE (0.16% / 0.12%).

The third row is measured differently and is the newest: that model has no MTP
head, so it is plain greedy decode with no speculation, 64 tokens at a 40k-token
depth, best of five reps, two rounds. Both arms run at ctx 262,144 with a q4_0 KV
cache; stock runs the options stock can run (`--n-cpu-moe 44`, ubatch 1024) and
the patched arm runs the ones the patches make reachable (`--n-cpu-moe 48`,
ubatch 6144 and the environment variables in
[docs/patches.md](docs/patches.md)) — on this model that *is* the win, since the
configuration is what the code buys. Prefill moves with it: a fresh 40k prompt
takes 418 s stock against **177 s** patched (96.4 → **227.4 t/s**, +136%). Of the
decode gain, patches 01–31 account for 8.63 → 9.87 t/s and 32–46 for the rest.
Stock is three rounds there, the patched arm two; the stock prefill spread
(423 / 413 / 403 s) is the host page cache warming to 41 GB of weights.

Both runs predate patches 29 and 30. The dense model is all Q4_1/Q5_1, so it is
unaffected; the MoE model carries 31.7% of its bytes in IQ3_XXS and 47.9% in
IQ2_XS, and picks up about 1% from patch 29 and 1.5% from patch 30 in separate
two-round A/Bs, so its figure is a slight underestimate.

**The table also predates the `v0.2.0` rebase**, and stock-vs-patched has not
been re-taken on `v0.2.0`. What was measured for the rebase is the upgrade
itself — the patched `b10133` build against the patched `v0.2.0` one, on the same
card with `llama-bench` (`-ngl 56 -fa 1`, arms interleaved, first round discarded,
three rounds each): 9B dense pp512 +1.15% / tg64 +0.15%, MoE pp512 −0.03% /
tg64 +0.47%. That bounds the rebase as a non-regression; it does not restate what
the set is worth against stock `v0.2.0`.

**The `v0.4.0` rebase adds a further gap**: neither the stock-vs-patched table
above nor the `v0.2.0` non-regression check has been re-taken against the tree
this rebase ships, so both numbers are now one more unmeasured rebase removed
from what actually runs.

**The table also predates patches 32–46**, and those were developed on a third
model, not on either of these two. Nine of the fifteen are off by default. Of the
six that are not, four fire only on one model architecture, one changes an MMQ
tile for IQ4_NL on no-DP4A cards, and one adds CUDA fusions whose matchers need a
hyper-connection graph — so on these two models the set should behave as it did
before. That was measured rather than assumed, on the third model the card still
has locally (27B dense, IQ3_XXS/IQ4_XS, all layers on the GPU, `llama-bench -fa 1
-ctk q4_0 -ctv q4_0`, six rounds with the arm order rotated): against the
published 31-patch set, pp512 138.23 → 138.47 t/s and tg64 22.84 → 22.84 t/s
(dropping each arm's first round, where a cold start costs both arms ~1%), and a
third arm with the new fusions disabled lands in the same place. No change at
this resolution.

This is the whole set against no patches. It is **not** the sum of the per-patch
numbers below, which were each measured against the stack as it stood at the
time and do not compose.

## Status

- All 44 patches are generated against llama.cpp **`v0.4.0`**, where they
  apply at **zero fuzz and zero offset** (`nix flake check` verifies both, and
  that the file list matches the ordered list in `nix/patches.nix`).
- Not submitted upstream. Two are straightforward candidates — 11
  `penalties-direct` and 21 `sched-reset-lazy` are architecture-independent,
  bit-identical, and fall back to the original path on any input they do not
  handle. 28 `top-k-partial` is CUDA-only instead: its `__shfl_xor_sync` call
  takes cub's 3-argument form, which the HIP vendor header maps to a
  4-argument macro, and its kernels assume a 32-lane warp throughout, so the
  whole block is guarded off with `#if !defined(GGML_USE_HIP)` and HIP builds
  keep using upstream's own fallback (its radix top-k above `ncols = 1024`, a
  full sort at or under it) untouched (see [docs/patches.md](docs/patches.md)).
  Nothing but time has kept any of them out.
- `test-backend-ops` on a P100 with the full set applied: **14,744 tests, and
  three full runs went 14,744/14,744 twice with one failing case once** — the
  same score as the 31-patch set on the same machine (also two clean runs of
  three), and an earlier 13,352-test run on `v0.2.0` matched its unpatched build.
  Read those counts as *no reproducible failures* rather than *never a red line*:
  the suite draws fresh random inputs every run, and
  `ADD(type=f16,ne=[10,5,4,3],nr=[2,1,1,1],nf=2)` fails intermittently on the
  **unpatched `v0.4.0` build as well** — 5 of 15 runs of the ADD subset there,
  2 of 15 with 31 patches, 1 of 15 with all 44 — so the single red line is a
  tolerance-borderline case, not something the patches introduce.
  `TOPK_MOE(ne=[288,22,1,1],n_expert_used=8,with_norm=0)` behaves the same way at
  a lower rate. Patches 32–46 additionally have per-patch evidence against the
  model they were written for (bit-identity checks, perplexity, a long-context
  benchmark), and a non-regression A/B on a third model, in
  [docs/patches.md](docs/patches.md).
- Unless a patch says otherwise, its output is **bit-identical** to the
  unpatched build. A few do change output (09 where K needs three of the four
  warps, 12 at the widths it takes, 15 by design, 31 once the KV cache is
  longer than its chunk length, 35 where a MUL_MAT_ID moves from MMQ to MMVQ,
  36 where a speculative verify batch stops going through cuBLAS, 39 by
  reduction order, 41 because its permutation is index-ascending where the sort
  it replaces was value-descending, 44 while the newest positions form a partial
  block, 45 when the padded flash-attention kernel is taken, 46 because an
  expert moves between the CPU and the GPU) and say so with the evidence. Of
  those, 44 and 45 are the two that are **on by default**. 22 is the one
  whose bit-identity is **measured rather than argued**: its `n_tokens <= 4`
  default comes from a dense and an MoE model staying identical there, not
  from a proof that a different compute-buffer layout cannot change a fusion
  decision. 19 and 20 are **off by default** and excluded from that default
  build: enabled, they are bit-identical on a dense model but produce
  non-deterministic MoE decode output (see docs/patches.md).
- **This repository is expected to shrink.** Anything upstream fixes should be
  deleted here rather than carried forward; the value is in the measurements as
  much as in the code.

## What's in it

Percentages are end-to-end throughput on the setup described below. Cells
giving µs, launch counts or ratios are **kernel-level only** — those patches
were individually at or below the end-to-end noise floor, and their entries in
[docs/patches.md](docs/patches.md) say so. The numbers are **not additive**:
each was measured against the stack as it stood at the time.

| # | Patch | Scope | Effect |
|---:|---|---|---|
| 01 | `vmad-dp4a-sm60` | sm_60 | +6.5–6.8% decode |
| 02 | `mmvq-rows-per-block-sm60` | pre-Turing | +23.0% (llama-bench tg32, Q4_0) |
| 04 | `concat-non-cont-flat` | CUDA | 18.0 → 4.7 µs kernel |
| 05 | `mmvf-f32-pascal` | pre-Turing | +3.7–4.3% |
| 07 | `mmvq-moe-rows-sm60` | all archs | +1.9% decode |
| 08 | `mmvq-mmid-batch-sm60` | pre-Volta | +2.2% |
| 09 | `mmvq-nwarps-small-k-sm60` | pre-Turing | +1.29% MoE decode |
| 10 | `mmvq-q8-1-activation-cache` | CUDA | +1.17% / +0.88%, ~0.5% less in this form |
| 11 | `penalties-direct` | host | +5.3% |
| 12 | `mmvq-f16-sm60` | sm_60 | +9.5% decode |
| 13 | `sampler-prefilter` | host | removes 2.2% of decode from the critical path |
| 14 | `getrows-narrow-rows` | CUDA | 278 µs → negligible |
| 15 | `mtp-draft-vocab` | model | +8.3% to −17.4%, set by coverage |
| 16 | `cpy-fastdiv` | CUDA | −56% kernel, +0.82% |
| 17 | `norm-register-cache` | CUDA | +0.71% / +0.95% |
| 18 | `fuse-sibling-nodes` | CUDA | +1.06% |
| 19 | `fuse-pre-add-rms-norm` | CUDA | +0.70%, **off by default** (`GGML_CUDA_FUSE_PRE_ADD=1`, see docs) |
| 20 | `fuse-add-unary-mul` | CUDA (delta-net) | +0.72%, **off by default** (`GGML_CUDA_FUSE_ADD_UNARY_MUL=1`, see docs) |
| 21 | `sched-reset-lazy` | host | +0.94% |
| 22 | `decode-sched-slots` | host | +0.97% (4 slots) |
| 23 | `fuse-gdn-beta-sigmoid` | CUDA (delta-net) | −4,080 launches |
| 24 | `fuse-gdn-state-gather` | CUDA (delta-net) | 1.4% of decode |
| 25 | `gdn-gather-single-snapshot` | CUDA (delta-net) | −7.3 ms of copies |
| 26 | `cpy-fused-rows` | CUDA | 1.9× kernel |
| 27 | `fuse-concat-gather` | CUDA (delta-net) | +0.74% |
| 28 | `top-k-partial` | CUDA | 7.6× kernel, ~1.1% of decode |
| 29 | `mmvq-iq3xxs-grid-smem` | CUDA | +3.1–7.6% decode (dense), bit-identical |
| 30 | `mmvq-ksigns-smem` | CUDA | +0.2–1.8% decode, −9.3% IQ3_XXS kernel, bit-identical |
| 31 | `fattn-f16-kv-chunk` | pre-Turing | −960 MiB compute buffer at ctx 262,144 (1,152 → 192), decode unchanged |
| 32 | `sched-split-prefetch` | host | −9–10% decode with offloaded MoE experts, **off by default** (`LLAMA_SCHED_PREFETCH=1`) |
| 33 | `sched-weight-prefetch` | host | −4.6–6.6% prefill, −1,222 MiB peak at ctx 262,144, **off by default** (`LLAMA_SCHED_WPF=<MiB>`) |
| 34 | `mul-mat-id-negative-ids` | host | enables 46; no effect on its own |
| 35 | `mmvq-mmid-batch-cap` | CUDA | −4–11% per ubatch with a resident expert cache, **off by default** (`LLAMA_MMVQ_MMID_MAX=<n>`) |
| 36 | `mmvq-chunk-large-batch` | CUDA | −1,166 MiB pool after a speculative batch, **off by default** (`GGML_CUDA_MMVQ_CHUNK_MIN_MIB=<MiB>`) |
| 37 | `getrows-narrow-batched` | CUDA | −9.4% prefill, bit-identical, **off by default** (`GGML_CUDA_GETROWS_FLAT_MAX=<n>`) |
| 38 | `getrows-q4-0-block` | CUDA | −10.1% prefill, bit-identical, **off by default** (`GGML_CUDA_GETROWS_Q4_0_BLK=1`) |
| 39 | `gdn-lanes-per-column` | CUDA (delta-net) | −4.3% prefill, −3% decode, **off by default** (`GGML_GDN_LPC=16`) |
| 40 | `mmq-iq4-nl-threads` | pre-Volta | +3–4% on an IQ4_NL MoE down projection, bit-identical |
| 41 | `top-k-radix-select` | CUDA | 133 → 74.8 µs at k = 2,051, −2.3% decode, **off by default** (`GGML_CUDA_TOP_K_SELECT=1`) |
| 42 | `qwen4exp-hc-exact` | model | −1% decode, bit-identical |
| 43 | `fuse-hc-combine` | CUDA | two kernels for ~40 element-wise ops per layer, bit-identical |
| 44 | `qwen4exp-qsa-block-key-cache` | model | removes decode's depth dependence (384 MiB at ctx 262,144) |
| 45 | `qwen4exp-qsa-sparse-gather` | model | compute buffer 3,773 → 1,063 MiB; −7.0% prefill and −3% decode with `LLAMA_QSA_PAD=1` |
| 46 | `qwen4exp-moe-expert-cache` | model | decode 51.0 ms/token at 40k on 60 slots (55.2 on 48), **off by default** (`LLAMA_MOE_CACHE=<slots>`) |

Patches 19 and 20 are off by default. Their per-patch figures above were
measured individually on a dense model; enabling both together is worth
**+2.36% decode on a 35B-A3B MoE** (same build, toggled only by the two
environment variables: 80.82 → 82.73 t/s, prompt processing unchanged).
That is the upper bound on what leaving them off costs, but it is measured
on exactly the configuration `docs/patches.md` reports as non-deterministic
on MoE graphs, so read the hazard note there before treating it as
throughput you can ship.

Scope tags, details, kill switches and the rejected alternatives:
**[docs/patches.md](docs/patches.md)**.

## Use it

### Nix

```nix
{
  inputs.llama-cpp-p100-patches.url = "github:shinbunbun/llama-cpp-p100-patches";
}
```

Either take the ordered patch list and apply it to your own build:

```nix
llama-cpp-patched = pkgs.llama-cpp.overrideAttrs (old: {
  patches = (old.patches or [ ]) ++ inputs.llama-cpp-p100-patches.lib.patches;
});
```

or use the overlay — apply it **last**, and assume it needs a pristine `v0.4.0`
tree, because zero-fuzz patches reject against anything that has already
rewritten the same lines:

```nix
nixpkgs.overlays = [ inputs.llama-cpp-p100-patches.overlays.default ];
```

`#llama-cpp-sm60` builds it for Pascal directly. There is no binary cache, so
that is a from-source CUDA build — hours, not minutes.

nixpkgs' default `cudaCapabilities` starts at 7.5, which leaves no sm_60 code in
the binary and fails at runtime with *"named symbol not found"*. Pascal builds
need `cudaCapabilities = [ "6.0" ]` and **CUDA 12.x** — CUDA 13 dropped Pascal
code generation, so the flake pins `cudaPackages_12`.

### Without Nix

```console
$ git clone --branch v0.4.0 https://github.com/ggml-org/llama.cpp
$ cd llama.cpp
$ for p in ../llama-cpp-p100-patches/patches/*.patch; do
    patch -p1 -F0 < "$p" || { echo "FAILED: $p"; break; }
  done
```

Apply in numeric order and **keep `-F0`**. Fuzzy application is the dangerous
failure mode here: it succeeds while silently landing a hunk in the wrong place.
Stop at the first failure rather than patching onto a half-patched tree.

The set is not all-or-nothing, but you cannot pick freely either — several
patches touch the same lines and later ones build on earlier ones. Dropping one
from the middle generally means rebasing the rest.

### Rebasing onto a newer llama.cpp

1. Bump `llamaCppTag` in `nix/patches.nix` **and** the `llama-cpp-src` input
   in `flake.nix` — they are separate literals and must be changed together.
2. `nix flake check`. It fails on exactly the patches that no longer apply.
3. For each failure decide which it is: **fixed upstream** — delete the patch,
   remove its row here and its entry in `docs/patches.md`, and say so; or
   **moved** — regenerate it against the new tree.
4. Re-run `test-backend-ops` and re-measure. A patch that still applies is not
   the same as a patch that is still worth having: several of these exist only
   because of a value upstream tuned for other hardware, and upstream may have
   retuned it.
5. Land the consumers in the same change. `lib.llamaCppTag` and
   `lib.upstreamTag` are the API: a consumer that reconstructs the tag from
   `llama-cpp.version` breaks silently the moment upstream changes its version
   scheme, which is what happened at `v0.2.0`. `lib.upstreamTag` returns `null`
   for a llama-cpp fetched by `rev`, so handle that rather than interpolating it.

## Where the numbers come from

- Tesla P100-PCIE-16GB (GP100, sm_60), CUDA 12.9, driver 580.x
- Qwen3.5 9B dense and Qwen3.6 35B-A3B (256 experts, 8 active, 41 layers), Q4_1 /
  Q5_1 / IQ-series weights
- Speculative decoding via MTP, so the hot path is a width-5 verify batch
- End-to-end throughput from `llama-server` over a fixed three-prompt set, with
  a realistic sampler (temperature 0.7, top-p 0.8, top-k 20) — sampler settings
  change conclusions here by several points, see the benchmarking notes
- `llama-batched-bench -npl 5` instead, for changes that alter the output, where
  a fixed prompt set is no longer comparable
- Later measurements report a two-prompt geometric mean: one of the three
  prompts turned out to be bimodal and to carry ~90% of the variance. Entries
  that predate that change say "three prompts"

The measurement discipline this needed — thermal control, resolving effects
below the noise floor, why isolated microbenchmarks mispredicted twice, and the
fusion traps — is written up in
**[docs/benchmarking.md](docs/benchmarking.md)**. That document is probably more
reusable than the patches.

## Caveats

- Tuned for **decode**, specifically single-stream decode with a speculative
  verify batch of 2–5. Large-batch serving was not a target.
- Patches scoped to `sm_60`, `pre-Volta` or `pre-Turing` change values upstream
  tuned for newer hardware, and `pre-Turing` includes Volta, which was not
  measured. Patch 07 is not gated at all. Measure before using any of them on
  Turing and later.
- `CUDA (delta-net)` patches only fire on models with a gated delta-net block
  (Qwen3-Next, Qwen3.5). Elsewhere they never fire; the only cost is the
  per-graph pattern match.
- `nix flake check` proves that the patches apply, that nixpkgs still carries the
  tag they were generated against, and that the patched tree compiles without
  CUDA. It does not compile the CUDA sources, run llama.cpp's tests, or measure
  anything — those need the card.

## Contributing

Issues and PRs welcome. The most useful contribution is a measurement on
hardware I do not have — P40, GTX 10xx, Maxwell, or anything Turing and later —
since every non-`sm_60` scope tag here is an inference from a single GP100.

## License

MIT, matching llama.cpp. See [LICENSE](LICENSE).
Unofficial, and unaffiliated with the llama.cpp project.
