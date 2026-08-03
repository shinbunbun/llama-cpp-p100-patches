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
      forAllSystems = f: lib.genAttrs systems (system: f system);

      patchSet = import ./nix/patches.nix;

      # nixpkgs' default cudaCapabilities start at 7.5 (Turing), which leaves no
      # sm_60 code in the binary and fails at runtime with "named symbol not
      # found".  CUDA 13 dropped Pascal codegen entirely, so 12.x is required.
      pkgsSm60 =
        system:
        import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
            cudaCapabilities = [ "6.0" ];
          };
          overlays = [ self.overlays.default ];
        };
    in
    {
      # Ordered list of patch files, for consumers that build llama-cpp themselves.
      #
      #   llama-cpp.overrideAttrs (old: {
      #     patches = (old.patches or []) ++ inputs.llama-cpp-perf-patches.lib.patches;
      #   })
      lib = { inherit (patchSet) patches llamaCppVersion; };

      overlays.default = import ./nix/overlay.nix;

      packages = forAllSystems (system: {
        default = self.packages.${system}.llama-cpp-sm60;
        llama-cpp-sm60 = (pkgsSm60 system).llama-cpp;
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
          patches-apply =
            pkgs.runCommand "patches-apply"
              {
                nativeBuildInputs = [ pkgs.patch ];
              }
              ''
                cp -r --no-preserve=mode ${llama-cpp-src} src
                cd src
                ${lib.concatMapStringsSep "\n" (p: ''
                  echo "applying ${builtins.baseNameOf p}"
                  patch -p1 -F0 --no-backup-if-mismatch < ${p}
                '') patchSet.patches}
                touch $out
              '';
        }
      );

      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt-tree);
    };
}
