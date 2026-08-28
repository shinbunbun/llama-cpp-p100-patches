# The upstream tag nixpkgs' llama-cpp actually fetches, or null when there is none.
#
# Upstream started cutting semver releases (v0.2.0) alongside the bNNNNN build
# tags in 2026-08, and nixpkgs followed: `version` became "0.2.0", so the tag
# can no longer be reconstructed from it.  `src.tag` is the tag itself and is
# correct under either scheme.
#
# fetchFromGitHub always defines `tag` and sets it to null when called with
# `rev`, so `src.tag or ...` never fires -- test the value, not its presence.
# Returning null rather than throwing is deliberate: a consumer comparing tags
# wants false, not an evaluation abort that takes its other outputs with it.
drv: drv.src.tag or null
