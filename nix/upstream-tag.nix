# The upstream tag nixpkgs' llama-cpp actually fetches.
#
# Upstream started cutting semver releases (v0.2.0) alongside the bNNNNN build
# tags in 2026-08, and nixpkgs followed: `version` became "0.2.0", so the tag
# can no longer be reconstructed from it.  `src.tag` is the tag itself and is
# correct under either scheme.
#
# fetchFromGitHub always defines `tag` and sets it to null when called with
# `rev`, so `src.tag or ...` never fires and would return null -- which only
# surfaces later as "cannot coerce null to a string" from whatever interpolates
# the result.  Test the value, not its presence.
drv:
let
  tag = drv.src.tag or null;
in
if tag == null then
  throw ''
    llama-cpp's src carries no tag: nixpkgs fetches it by rev, or the src was
    overridden.  The patches are generated against a tag and cannot be checked
    against a bare revision -- pin nixpkgs to a llama-cpp built from a tag.
  ''
else
  tag
