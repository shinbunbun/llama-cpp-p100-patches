# Appends the patch set to nixpkgs' llama-cpp.
#
# The overlay does not pin the llama.cpp version.  Point nixpkgs at the tag in
# nix/patches.nix, or apply the patches yourself if you need a different one.
final: prev: {
  llama-cpp = prev.llama-cpp.overrideAttrs (old: {
    patches = (old.patches or [ ]) ++ (import ./patches.nix).patches;
  });
}
