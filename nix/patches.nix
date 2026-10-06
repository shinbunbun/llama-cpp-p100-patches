# The patch set, in application order.
#
# Order matters: several patches touch the same files, and later ones build on
# earlier ones (for example 09 reinterprets a value that 02 introduced).
#
# `llamaCppTag` is the upstream tag the patches are generated against.  They
# apply at zero fuzz and zero offset on that tag; `nix flake check` verifies
# that, and that this list matches the contents of patches/.
{
  # Keep in sync with inputs.llama-cpp-src.url in flake.nix -- flake inputs must
  # be literals, so the tag cannot be shared between the two.
  llamaCppTag = "v0.5.0";

  patches = [
    ../patches/01-vmad-dp4a-sm60.patch
    ../patches/02-mmvq-rows-per-block-sm60.patch
    ../patches/04-concat-non-cont-flat.patch
    ../patches/05-mmvf-f32-pascal.patch
    ../patches/07-mmvq-moe-rows-sm60.patch
    ../patches/08-mmvq-mmid-batch-sm60.patch
    ../patches/09-mmvq-nwarps-small-k-sm60.patch
    ../patches/10-mmvq-q8-1-activation-cache.patch
    ../patches/11-penalties-direct.patch
    ../patches/12-mmvq-f16-sm60.patch
    ../patches/13-sampler-prefilter.patch
    ../patches/14-getrows-narrow-rows.patch
    ../patches/15-mtp-draft-vocab.patch
    ../patches/16-cpy-fastdiv.patch
    ../patches/17-norm-register-cache.patch
    ../patches/18-fuse-sibling-nodes.patch
    ../patches/19-fuse-pre-add-rms-norm.patch
    ../patches/20-fuse-add-unary-mul.patch
    ../patches/21-sched-reset-lazy.patch
    ../patches/22-decode-sched-slots.patch
    ../patches/23-fuse-gdn-beta-sigmoid.patch
    ../patches/24-fuse-gdn-state-gather.patch
    ../patches/25-gdn-gather-single-snapshot.patch
    ../patches/26-cpy-fused-rows.patch
    ../patches/27-fuse-concat-gather.patch
    ../patches/28-top-k-partial.patch
    ../patches/29-mmvq-iq3xxs-grid-smem.patch
    ../patches/30-mmvq-ksigns-smem.patch
    ../patches/31-fattn-f16-kv-chunk.patch
    ../patches/32-sched-split-prefetch.patch
    ../patches/33-sched-weight-prefetch.patch
    ../patches/34-mul-mat-id-negative-ids.patch
    ../patches/35-mmvq-mmid-batch-cap.patch
    ../patches/36-mmvq-chunk-large-batch.patch
    ../patches/37-getrows-narrow-batched.patch
    ../patches/38-getrows-q4-0-block.patch
    ../patches/39-gdn-lanes-per-column.patch
    ../patches/40-mmq-iq4-nl-threads.patch
    ../patches/41-top-k-radix-select.patch
    ../patches/42-qwen4exp-hc-exact.patch
    ../patches/44-qwen4exp-qsa-block-key-cache.patch
    ../patches/45-qwen4exp-qsa-sparse-gather.patch
    ../patches/46-qwen4exp-moe-expert-cache.patch
    ../patches/47-sched-split-inputs-cap.patch
    ../patches/48-fuse-rms-norm-scale.patch
  ];
}
