# The upstream tag nixpkgs' llama-cpp actually fetches.
#
# Upstream started cutting semver releases (v0.2.0) alongside the bNNNNN build
# tags in 2026-08, and nixpkgs followed: `version` became "0.2.0", so the tag
# can no longer be reconstructed from it.  `src.tag` is the tag itself and is
# correct under either scheme.
drv:
drv.src.tag or (throw ''
  llama-cpp's src has no `tag` attribute; nixpkgs may have switched
  fetchFromGitHub back to `rev`.  Fix nix/upstream-tag.nix rather than
  comparing `version`, which does not carry the tag name.
'')
