{
  description = "llama.cpp performance patches, developed and measured on Tesla P100 (GP100, sm_60)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # Pinned only so `nix flake check` can verify the patches still apply at zero
    # fuzz.  Consumers are not forced onto this revision.
    llama-cpp-src = {
      url = "github:ggml-org/llama.cpp/b10133";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      llama-cpp-src,
    }:
    let
      inherit (nixpkgs) lib;
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = lib.genAttrs systems;

      patchSet = import ./nix/patches.nix;

      # nixpkgs' default cudaCapabilities start at 7.5 (Turing), which leaves no
      # sm_60 code in the binary and fails at runtime with "named symbol not
      # found".  CUDA 13 dropped Pascal codegen entirely, so 12.x is pinned
      # explicitly rather than left to nixpkgs' default, which will move.
      pkgsSm60 =
        system:
        import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
            cudaCapabilities = [ "6.0" ];
          };
          overlays = [
            (_: prev: { cudaPackages = prev.cudaPackages_12; })
            self.overlays.default
          ];
        };
    in
    {
      # Ordered list of patch files, for consumers that build llama-cpp themselves.
      #
      #   llama-cpp.overrideAttrs (old: {
      #     patches = (old.patches or []) ++ inputs.llama-cpp-p100-patches.lib.patches;
      #   })
      lib = { inherit (patchSet) patches llamaCppVersion; };

      overlays.default = import ./nix/overlay.nix;

      # sm_60 only ever shipped on x86_64 hosts: Pascal never reached ARM servers,
      # and the Pascal Jetsons are sm_53 / sm_62.  lib.patches and the overlay are
      # system-independent, so other platforms can still use those.
      packages = lib.genAttrs [ "x86_64-linux" ] (system: {
        default = self.packages.${system}.llama-cpp-sm60;
        llama-cpp-sm60 = (pkgsSm60 system).llama-cpp.overrideAttrs (old: {
          pname = "llama-cpp-sm60";
        });
      });

      checks = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          # Every patch must apply to the pinned tag with no fuzz.  Fuzzy
          # application is the dangerous failure mode: it succeeds while silently
          # landing a hunk in the wrong place.
          # Three things must hold, and only the first is obvious:
          #   1. every patch applies to the pinned tag
          #   2. with no fuzz AND no offset -- -F0 stops fuzzy context matching but
          #      still lets a hunk relocate, and mmvq.cu is full of near-identical
          #      per-quantization blocks for it to relocate into
          #   3. nix/patches.nix lists exactly the files in patches/ -- the list
          #      carries the order, so it cannot be globbed, but it can go stale
          patches-apply = pkgs.runCommand "patches-apply" { } ''
            listed=${
              pkgs.lib.escapeShellArg (pkgs.lib.concatMapStringsSep "\n" builtins.baseNameOf patchSet.patches)
            }
            onDisk=${
              pkgs.lib.escapeShellArg (
                pkgs.lib.concatStringsSep "\n" (
                  pkgs.lib.naturalSort (
                    builtins.attrNames (
                      pkgs.lib.filterAttrs (n: t: t == "regular" && pkgs.lib.hasSuffix ".patch" n) (
                        builtins.readDir ./patches
                      )
                    )
                  )
                )
              )
            }
            if [ "$listed" != "$onDisk" ]; then
              echo "nix/patches.nix does not match patches/:"
              diff <(echo "$listed") <(echo "$onDisk") || true
              exit 1
            fi

            cp -r --no-preserve=mode ${llama-cpp-src} src
            cd src
            ${lib.concatMapStringsSep "\n" (p: ''
              echo "applying ${builtins.baseNameOf p}"
              log=$(patch -p1 -F0 --batch --no-backup-if-mismatch < ${p})
              echo "$log"
              if grep -q offset <<< "$log"; then
                echo "ERROR: ${builtins.baseNameOf p} applied at an offset; regenerate it"
                exit 1
              fi
            '') patchSet.patches}
            touch $out
          '';
        }
      );

      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt-tree);
    };
}
