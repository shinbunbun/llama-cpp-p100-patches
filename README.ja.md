# llama-cpp-perf-patches

[llama.cpp](https://github.com/ggml-org/llama.cpp) の性能パッチ 28 本。
**Tesla P100 (GP100, sm_60)** 上で開発・実測した。

English: [README.md](README.md)

出発点は Pascal である。DP4A を持たないため llama.cpp の量子化 matmul 経路は
エミュレーションと cuBLAS に落ちる。全体の約 1/3 は、Pascal が代わりに持っている
もの (full-rate な HFMA2) を使い切るためのパッチである。

ただし大半は Pascal 固有ではない。カーネル融合、添字計算の修正、ホスト側の
サンプラー経路 2 本、そして語彙全体のソートを避ける `top_k` は、どの GPU でも効く。

すべて実測に基づく。**棄却した代案も含めて**、根拠は各パッチがソースに追加する
コメントに書いてある。

## 状態

- llama.cpp **`b10133`** に対して **fuzz 0 で適用できる** (`nix flake check` が検証)
- upstream には未提出。報告価値が高いものは
  [docs/patches.ja.md](docs/patches.ja.md) で明示している。それ以外は
  upstream が回帰テストできないハードウェアに意図的に限定してある
- 明記がない限り、出力は無パッチ版と**ビット一致**する。例外はパッチごとに
  perplexity 付きで書いてある

## 内容

| # | パッチ | 適用範囲 | 効果 |
|---:|---|---|---|
| 01 | `vmad-dp4a-sm60` | sm_60 | decode +6.5〜6.8% |
| 02 | `mmvq-rows-per-block-sm60` | pre-Turing | decode +15.4〜16.5% |
| 03 | `topk-moe-multirow` | CUDA | decode +2.8〜6.1% |
| 04 | `concat-non-cont-flat` | CUDA | カーネル 18.0 → 4.7 µs |
| 05 | `mmvf-f32-pascal` | pre-Turing | +3.7〜3.9% |
| 06 | `mmq-mul-mat-id-sm60` | sm_60 | MoE prefill +20〜41%、VRAM −200 MiB |
| 07 | `mmvq-moe-rows-sm60` | pre-Turing | decode +1.9% |
| 08 | `mmvq-mmid-batch-sm60` | sm_60 | +2.2% |
| 09 | `mmvq-nwarps-small-k-sm60` | pre-Turing | MoE decode +1.29% |
| 10 | `mmvq-q8-1-activation-cache` | CUDA | +1.17% / +0.88% |
| 11 | `penalties-direct` | host | +5.3% |
| 12 | `mmvq-f16-sm60` | sm_60 | decode +9.5% |
| 13 | `sampler-prefilter` | host | decode の 2.2% をクリティカルパスから外す |
| 14 | `getrows-narrow-rows` | CUDA | 278 µs → 無視できる量 |
| 15 | `mtp-draft-vocab` | model | 最大 +8.3% (被覆率依存) |
| 16 | `cpy-fastdiv` | CUDA | カーネル −56%、+0.82% |
| 17 | `norm-register-cache` | CUDA | +0.71% / +0.95% |
| 18 | `fuse-sibling-nodes` | CUDA | +1.06% |
| 19 | `fuse-pre-add-rms-norm` | CUDA | +0.70% |
| 20 | `fuse-add-unary-mul` | CUDA | +0.72% |
| 21 | `sched-reset-lazy` | host | +0.94% |
| 22 | `decode-sched-slots` | host | +0.97% (枠 4) |
| 23 | `fuse-gdn-beta-sigmoid` | CUDA | 起動 −4,080 発 |
| 24 | `fuse-gdn-state-gather` | CUDA | decode の 1.4% |
| 25 | `gdn-gather-single-snapshot` | CUDA | コピー −7.3 ms |
| 26 | `cpy-fused-rows` | CUDA | カーネル 1.9 倍 |
| 27 | `fuse-concat-gather` | CUDA | +0.74% |
| 28 | `top-k-partial` | CUDA | カーネル 7.6 倍、decode の約 1.1% |

詳細・停止スイッチ・棄却した代案は
**[docs/patches.ja.md](docs/patches.ja.md)** を参照。

パーセントは、セルにカーネル名が書いてある場合を除き、下記の環境における
end-to-end のスループット。**加算はできない** — それぞれ、その時点のスタックに
対して測った値である。

## 使い方

### Nix

```nix
{
  inputs.llama-cpp-perf-patches.url = "github:shinbunbun/llama-cpp-perf-patches";
}
```

順序付きのパッチ列を自分のビルドに適用するか:

```nix
llama-cpp-patched = pkgs.llama-cpp.overrideAttrs (old: {
  patches = (old.patches or [ ]) ++ inputs.llama-cpp-perf-patches.lib.patches;
});
```

overlay を使う:

```nix
nixpkgs.overlays = [ inputs.llama-cpp-perf-patches.overlays.default ];
```

Pascal 向けのビルド済みパッケージもある:

```console
$ nix build github:shinbunbun/llama-cpp-perf-patches#llama-cpp-sm60
```

`nixpkgs` の `cudaCapabilities` は既定が 7.5 以上なので、そのままでは生成物に
sm_60 コードが無く、実行時に *"named symbol not found"* で落ちる。Pascal 向けには
`cudaCapabilities = [ "6.0" ]` と **CUDA 12.x** が要る (CUDA 13 は Pascal の
コード生成を削除している)。

### Nix 以外

```console
$ git clone --branch b10133 https://github.com/ggml-org/llama.cpp
$ cd llama.cpp
$ for p in ../llama-cpp-perf-patches/patches/*.patch; do patch -p1 -F0 < "$p"; done
```

番号順に適用し、**`-F0` を外さないこと**。ここで危険なのは reject ではなく
**誤適用**である。fuzz を許すと、適用に成功したまま hunk が別の場所に入る。

全部入れる必要はないが、自由に選べるわけでもない。複数のパッチが同じ行に触り、
後のパッチが前のパッチの上に成り立っている。途中の 1 本を抜くと、通常は残りの
リベースが必要になる。

## 測定環境

- Tesla P100-PCIE-16GB (GP100, sm_60)、CUDA 12.9
- Qwen3.5 9B dense と Qwen3.5-MoE、Q4_1 / Q5_1 / IQ 系の重み
- MTP による投機デコード。したがってホットパスは幅 5 の検証バッチ
- 固定プロンプト集合に対する `llama-server` の end-to-end スループット、または
  出力が壊れる変更では `llama-batched-bench -npl 5`

これに必要だった測定の作法 — 熱の制御、ノイズ下限より小さい効果の測り方、
隔離マイクロベンチが 2 回続けて外した話、融合の罠 — は
**[docs/benchmarking.ja.md](docs/benchmarking.ja.md)** にまとめてある。
おそらくパッチ本体より再利用が効く。

## 注意

- **decode 向け**、特に投機デコードの検証バッチ幅 2〜5 の単一ストリーム decode に
  合わせてある。大バッチのサービングは対象外
- `sm_60` / `pre-Turing` のパッチは、upstream が新しいハードウェア向けに調整した
  値を変更している。Turing 以降で使う前に必ず測ること
- 一部の融合は gated delta-net のモデル (Qwen3-Next / Qwen3.5) でしか発火しない。
  他のアーキテクチャでは無効になるだけで、害はない
- `nix flake check` が保証するのは「パッチが当たること」だけで、llama.cpp の
  テストは走らせていない。リベース後は `test-backend-ops` を回すこと

## ライセンス

llama.cpp に合わせて MIT。[LICENSE](LICENSE) を参照。
