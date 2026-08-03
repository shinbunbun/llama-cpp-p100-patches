# llama-cpp-perf-patches

28 performance patches for [llama.cpp](https://github.com/ggml-org/llama.cpp),
developed and measured on a **Tesla P100 (GP100, sm_60)**.

日本語版: [README.ja.md](README.ja.md)

Pascal is the reason this exists — it has no DP4A, so llama.cpp's quantized
matmul paths fall back to emulation and cuBLAS, and roughly a third of the
patches exploit what Pascal *does* have (full-rate HFMA2). But most of the set
is not Pascal-specific: kernel fusions, index-arithmetic fixes, two host-side
sampler paths, and a `top_k` that avoids sorting the whole vocabulary all apply
anywhere.

Everything here was measured, and the measurements — including the rejected
alternatives — are in the comments each patch adds to the source.

## Status

- Against llama.cpp **`b10133`**; they apply at **zero fuzz** (`nix flake check`
  verifies this).
- Not submitted upstream. Some are good candidates and are marked as such in
  [docs/patches.md](docs/patches.md); others are deliberately scoped to
  hardware that upstream cannot regression-test.
- Unless stated otherwise, output is **bit-identical** to the unpatched build.
  The exceptions are called out per patch, with perplexity measurements.

## What's in it

| # | Patch | Scope | Effect |
|---:|---|---|---|
| 01 | `vmad-dp4a-sm60` | sm_60 | +6.5–6.8% decode |
| 02 | `mmvq-rows-per-block-sm60` | pre-Turing | +15.4–16.5% decode |
| 03 | `topk-moe-multirow` | CUDA | +2.8–6.1% decode |
| 04 | `concat-non-cont-flat` | CUDA | 18.0 → 4.7 µs kernel |
| 05 | `mmvf-f32-pascal` | pre-Turing | +3.7–3.9% |
| 06 | `mmq-mul-mat-id-sm60` | sm_60 | MoE prefill +20–41%, −200 MiB VRAM |
| 07 | `mmvq-moe-rows-sm60` | pre-Turing | +1.9% decode |
| 08 | `mmvq-mmid-batch-sm60` | sm_60 | +2.2% |
| 09 | `mmvq-nwarps-small-k-sm60` | pre-Turing | +1.29% MoE decode |
| 10 | `mmvq-q8-1-activation-cache` | CUDA | +1.17% / +0.88% |
| 11 | `penalties-direct` | host | +5.3% |
| 12 | `mmvq-f16-sm60` | sm_60 | +9.5% decode |
| 13 | `sampler-prefilter` | host | 2.2% of decode, off the critical path |
| 14 | `getrows-narrow-rows` | CUDA | 278 µs → negligible |
| 15 | `mtp-draft-vocab` | model | up to +8.3%, coverage-dependent |
| 16 | `cpy-fastdiv` | CUDA | −56% kernel, +0.82% |
| 17 | `norm-register-cache` | CUDA | +0.71% / +0.95% |
| 18 | `fuse-sibling-nodes` | CUDA | +1.06% |
| 19 | `fuse-pre-add-rms-norm` | CUDA | +0.70% |
| 20 | `fuse-add-unary-mul` | CUDA | +0.72% |
| 21 | `sched-reset-lazy` | host | +0.94% |
| 22 | `decode-sched-slots` | host | +0.97% (4 slots) |
| 23 | `fuse-gdn-beta-sigmoid` | CUDA | −4,080 launches |
| 24 | `fuse-gdn-state-gather` | CUDA | 1.4% of decode |
| 25 | `gdn-gather-single-snapshot` | CUDA | −7.3 ms of copies |
| 26 | `cpy-fused-rows` | CUDA | 1.9× kernel |
| 27 | `fuse-concat-gather` | CUDA | +0.74% |
| 28 | `top-k-partial` | CUDA | 7.6× kernel, ~1.1% of decode |

Details, kill switches and the rejected alternatives:
**[docs/patches.md](docs/patches.md)**.

Percentages are end-to-end throughput on the setup described below unless the
cell names a kernel. They are not additive — each was measured against the
stack as it stood at the time.

## Use it

### Nix

```nix
{
  inputs.llama-cpp-perf-patches.url = "github:shinbunbun/llama-cpp-perf-patches";
}
```

Either take the ordered patch list and apply it to your own build:

```nix
llama-cpp-patched = pkgs.llama-cpp.overrideAttrs (old: {
  patches = (old.patches or [ ]) ++ inputs.llama-cpp-perf-patches.lib.patches;
});
```

or use the overlay:

```nix
nixpkgs.overlays = [ inputs.llama-cpp-perf-patches.overlays.default ];
```

There is also a ready-made Pascal build:

```console
$ nix build github:shinbunbun/llama-cpp-perf-patches#llama-cpp-sm60
```

`nixpkgs` defaults `cudaCapabilities` to 7.5 and up, which leaves no sm_60 code
in the binary and fails at runtime with *"named symbol not found"*. Pascal
builds need `cudaCapabilities = [ "6.0" ]` and **CUDA 12.x** — CUDA 13 dropped
Pascal code generation.

### Without Nix

```console
$ git clone --branch b10133 https://github.com/ggml-org/llama.cpp
$ cd llama.cpp
$ for p in ../llama-cpp-perf-patches/patches/*.patch; do patch -p1 -F0 < "$p"; done
```

Apply in numeric order and **keep `-F0`**. Fuzzy application is the dangerous
failure mode here: it succeeds while silently landing a hunk in the wrong place.

The set is not all-or-nothing, but it is not free-choice either — several
patches touch the same lines and later ones build on earlier ones. Dropping one
from the middle generally means rebasing the rest.

## Where the numbers come from

- Tesla P100-PCIE-16GB (GP100, sm_60), CUDA 12.9
- Qwen3.5 9B dense and a Qwen3.5-MoE, Q4_1 / Q5_1 / IQ-series weights
- Speculative decoding via MTP, so the hot path is a width-5 verify batch
- End-to-end throughput from `llama-server` over a fixed prompt set, or
  `llama-batched-bench -npl 5` where a change breaks output

The measurement discipline this needed — thermal control, resolving effects
below the noise floor, why isolated microbenchmarks mispredicted twice, and the
fusion traps — is written up in
**[docs/benchmarking.md](docs/benchmarking.md)**. That document is probably more
reusable than the patches.

## Caveats

- Tuned for **decode**, specifically single-stream decode with a speculative
  verify batch of 2–5. Large-batch serving was not a target.
- Patches marked `sm_60` or `pre-Turing` change values that upstream tuned for
  newer hardware. Measure before using them on Turing and later.
- Some fusions only fire for gated delta-net models (Qwen3-Next, Qwen3.5). On
  other architectures they are inert, not harmful.
- `nix flake check` proves the patches *apply*; it does not run llama.cpp's
  tests. Run `test-backend-ops` after any rebase.

## License

MIT, matching llama.cpp. See [LICENSE](LICENSE).
