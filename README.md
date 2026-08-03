# llama-cpp-p100-patches

28 performance patches for [llama.cpp](https://github.com/ggml-org/llama.cpp),
developed and measured on a **Tesla P100 (GP100, sm_60)**.

日本語版: [README.ja.md](README.ja.md)

```console
$ nix build github:shinbunbun/llama-cpp-p100-patches#llama-cpp-sm60
```

GP100 is the reason this exists. Alone among Pascal chips it has no DP4A and
has full-rate HFMA2 — so llama.cpp's quantized matmul paths fall back to
emulation and cuBLAS, while the one instruction the card is genuinely good at
goes unused. The patches named `sm60` exploit that asymmetry.

**But only 7 of the 28 are gated to Pascal-era hardware.** One more changes an
unconditional constant that every GPU sees. The remaining 20 are not
hardware-scoped at all — kernel fusions, index-arithmetic fixes, two host-side
sampler paths, two scheduler patches, and a `top_k` that avoids sorting the
whole vocabulary — though five of those only fire on gated delta-net models and
one needs a specific model feature. Every row in the table below carries a scope
tag, and [docs/patches.md](docs/patches.md) is grouped by it.

Every patch has a measurement behind it, and the measurements — including the
alternatives that were tried and rejected — are in the comments the patches add
to the source.

## Status

- Generated against llama.cpp **`b10133`**, where they apply at **zero fuzz and
  zero offset** (`nix flake check` verifies both, and that the file list matches
  the ordered list in `nix/patches.nix`).
- Not submitted upstream. Three are straightforward candidates — 11
  `penalties-direct`, 21 `sched-reset-lazy`, 28 `top-k-partial` are all
  architecture-independent, bit-identical, and fall back to the original path on
  any input they do not handle. Nothing but time has kept them out.
- Unless a patch says otherwise, its output is **bit-identical** to the
  unpatched build. Three do change output (03 above one row, 12 at the widths it
  takes, 15 by design) and say so with the evidence.
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
| 03 | `topk-moe-multirow` | CUDA | +2.8–6.1% decode |
| 04 | `concat-non-cont-flat` | CUDA | 18.0 → 4.7 µs kernel |
| 05 | `mmvf-f32-pascal` | pre-Turing | +3.7–4.3% |
| 06 | `mmq-mul-mat-id-sm60` | sm_60 | MoE prefill +20–41%, −200 MiB VRAM |
| 07 | `mmvq-moe-rows-sm60` | all archs | +1.9% decode |
| 08 | `mmvq-mmid-batch-sm60` | pre-Volta | +2.2% |
| 09 | `mmvq-nwarps-small-k-sm60` | pre-Turing | +1.29% MoE decode |
| 10 | `mmvq-q8-1-activation-cache` | CUDA | +1.17% / +0.88% |
| 11 | `penalties-direct` | host | +5.3% |
| 12 | `mmvq-f16-sm60` | sm_60 | +9.5% decode |
| 13 | `sampler-prefilter` | host | removes 2.2% of decode from the critical path |
| 14 | `getrows-narrow-rows` | CUDA | 278 µs → negligible |
| 15 | `mtp-draft-vocab` | model | +8.3% to −17.4%, set by coverage |
| 16 | `cpy-fastdiv` | CUDA | −56% kernel, +0.82% |
| 17 | `norm-register-cache` | CUDA | +0.71% / +0.95% |
| 18 | `fuse-sibling-nodes` | CUDA | +1.06% |
| 19 | `fuse-pre-add-rms-norm` | CUDA | +0.70% |
| 20 | `fuse-add-unary-mul` | CUDA (delta-net) | +0.72% |
| 21 | `sched-reset-lazy` | host | +0.94% |
| 22 | `decode-sched-slots` | host | +0.97% (4 slots) |
| 23 | `fuse-gdn-beta-sigmoid` | CUDA (delta-net) | −4,080 launches |
| 24 | `fuse-gdn-state-gather` | CUDA (delta-net) | 1.4% of decode |
| 25 | `gdn-gather-single-snapshot` | CUDA (delta-net) | −7.3 ms of copies |
| 26 | `cpy-fused-rows` | CUDA | 1.9× kernel |
| 27 | `fuse-concat-gather` | CUDA (delta-net) | +0.74% |
| 28 | `top-k-partial` | CUDA | 7.6× kernel, ~1.1% of decode |

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

or use the overlay — apply it **last**, and assume it needs a pristine `b10133`
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
$ git clone --branch b10133 https://github.com/ggml-org/llama.cpp
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

1. Bump `llamaCppVersion` in `nix/patches.nix` **and** the `llama-cpp-src` input
   in `flake.nix` — they are separate literals and must be changed together.
2. `nix flake check`. It fails on exactly the patches that no longer apply.
3. For each failure decide which it is: **fixed upstream** — delete the patch,
   remove its row here and its entry in `docs/patches.md`, and say so; or
   **moved** — regenerate it against the new tree.
4. Re-run `test-backend-ops` and re-measure. A patch that still applies is not
   the same as a patch that is still worth having: several of these exist only
   because of a value upstream tuned for other hardware, and upstream may have
   retuned it.

## Where the numbers come from

- Tesla P100-PCIE-16GB (GP100, sm_60), CUDA 12.9, driver 580.x
- Qwen3.5 9B dense and a 256-expert Qwen3.5-MoE (8 active, 41 layers), Q4_1 /
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
- `nix flake check` proves the patches apply. It does not run llama.cpp's tests.

## Contributing

Issues and PRs welcome. The most useful contribution is a measurement on
hardware I do not have — P40, GTX 10xx, Maxwell, or anything Turing and later —
since every non-`sm_60` scope tag here is an inference from a single GP100.

## License

MIT, matching llama.cpp. See [LICENSE](LICENSE).
Unofficial, and unaffiliated with the llama.cpp project.
