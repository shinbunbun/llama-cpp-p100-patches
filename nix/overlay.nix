# Appends the patch set to nixpkgs' llama-cpp.
#
# Idempotent, and warns rather than asserts when nixpkgs' llama-cpp is not the
# tag the patches were generated against: the patches apply at zero fuzz, so a
# mismatch rejects loudly rather than misapplying, and someone on an adjacent
# revision may reasonably want to try.
final: prev:
let
  patchSet = import ./patches.nix;
  expected = patchSet.llamaCppVersion;
  actual = "b" + prev.llama-cpp.version;
in
{
  llama-cpp =
    if prev.llama-cpp.passthru.p100Patched or false then
      prev.llama-cpp
    else
      prev.lib.warnIf (actual != expected)
        ''
          llama-cpp-p100-patches: the patches are generated against ${expected}, but
          nixpkgs has ${actual}.  They apply at zero fuzz only on ${expected} and will
          reject otherwise.  Pin nixpkgs to that tag, or rebase the patches.
        ''
        (
          prev.llama-cpp.overrideAttrs (old: {
            patches = (old.patches or [ ]) ++ patchSet.patches;
            passthru = (old.passthru or { }) // {
              p100Patched = true;
            };
          })
        );
}
