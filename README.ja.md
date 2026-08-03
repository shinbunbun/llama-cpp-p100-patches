# llama-cpp-p100-patches

[llama.cpp](https://github.com/ggml-org/llama.cpp) の性能パッチ 28 本。
**Tesla P100 (GP100, sm_60)** 上で開発・実測した。

English: [README.md](README.md)

```console
$ nix build github:shinbunbun/llama-cpp-p100-patches#llama-cpp-sm60
```

出発点は GP100 である。Pascal の中でこのチップだけが DP4A を持たず、代わりに
フルレートの HFMA2 を持つ。そのため llama.cpp の量子化 matmul 経路はエミュレーションと
cuBLAS に落ちる一方、このカードが本当に速い命令は使われないままになる。名前に
`sm60` が付くパッチはこの非対称性を突くものである。

**ただし 28 本のうち Pascal 世代に限定されているのは 7 本だけ**である。もう 1 本は
アーキテクチャで分岐しない定数を変えるので全 GPU に掛かる。残り 20 本はハードウェアに
依存しない — カーネル融合、添字計算の修正、ホスト側のサンプラー経路 2 本、スケジューラ
2 本、語彙全体のソートを避ける `top_k`。ただしそのうち 5 本は gated delta-net の
モデルでしか発火せず、1 本は特定のモデル機能を要求する。下の表の各行に適用範囲タグが
あり、[docs/patches.ja.md](docs/patches.ja.md) はその分類で章立てしてある。

すべてのパッチに実測の裏付けがある。**試して棄却した代案も含めて**、根拠は各パッチが
ソースに追加するコメントに書いてある。

## 状態

- llama.cpp **`b10133`** に対して生成しており、**fuzz 0・オフセット 0** で適用できる
  (`nix flake check` が両方と、`nix/patches.nix` の順序付きリストが `patches/` の
  内容と一致することを検証する)
- upstream には未提出。明快な候補が 3 本ある — 11 `penalties-direct` /
  21 `sched-reset-lazy` / 28 `top-k-partial` はいずれもアーキテクチャ非依存、
  ビット一致、扱えない入力では元の経路にフォールバックする。出していないのは
  時間の問題でしかない
- P100 上で全パッチ適用のまま **`test-backend-ops` が通る**: 13,329 件・失敗 0 を
  3 回連続。無パッチ版と同じ結果である
- パッチ自身が断らない限り、出力は無パッチ版と**ビット一致**する。出力が変わるのは
  3 本 (03 は 1 行を超える場合、12 は通る幅で、15 は設計上) で、それぞれ根拠付きで
  明示してある
- **このリポジトリは縮んでいくことを前提としている。** upstream が直したものは
  ここから消すべきで、価値はコードと同じくらい測定値の側にある

## 内容

パーセントは下記環境での end-to-end スループット。µs・起動回数・倍率を書いてあるセルは
**カーネル単体の値**であり、それらのパッチは end-to-end のノイズ下限以下だった
(各項目にそう書いてある)。**加算はできない** — それぞれ、その時点のスタックに対して
測った値である。

| # | パッチ | 適用範囲 | 効果 |
|---:|---|---|---|
| 01 | `vmad-dp4a-sm60` | sm_60 | decode +6.5〜6.8% |
| 02 | `mmvq-rows-per-block-sm60` | pre-Turing | +23.0% (llama-bench tg32, Q4_0) |
| 03 | `topk-moe-multirow` | CUDA | decode +2.8〜6.1% |
| 04 | `concat-non-cont-flat` | CUDA | カーネル 18.0 → 4.7 µs |
| 05 | `mmvf-f32-pascal` | pre-Turing | +3.7〜4.3% |
| 06 | `mmq-mul-mat-id-sm60` | sm_60 | MoE prefill +20〜41%、VRAM −200 MiB |
| 07 | `mmvq-moe-rows-sm60` | all archs | decode +1.9% |
| 08 | `mmvq-mmid-batch-sm60` | pre-Volta | +2.2% |
| 09 | `mmvq-nwarps-small-k-sm60` | pre-Turing | MoE decode +1.29% |
| 10 | `mmvq-q8-1-activation-cache` | CUDA | +1.17% / +0.88%, この形では約 0.5% 少ない |
| 11 | `penalties-direct` | host | +5.3% |
| 12 | `mmvq-f16-sm60` | sm_60 | decode +9.5% |
| 13 | `sampler-prefilter` | host | decode の 2.2% をクリティカルパスから外す |
| 14 | `getrows-narrow-rows` | CUDA | 278 µs → 無視できる量 |
| 15 | `mtp-draft-vocab` | model | 被覆率次第で +8.3% 〜 −17.4% |
| 16 | `cpy-fastdiv` | CUDA | カーネル −56%、+0.82% |
| 17 | `norm-register-cache` | CUDA | +0.71% / +0.95% |
| 18 | `fuse-sibling-nodes` | CUDA | +1.06% |
| 19 | `fuse-pre-add-rms-norm` | CUDA | +0.70% |
| 20 | `fuse-add-unary-mul` | CUDA (delta-net) | +0.72% |
| 21 | `sched-reset-lazy` | host | +0.94% |
| 22 | `decode-sched-slots` | host | +0.97% (枠 4) |
| 23 | `fuse-gdn-beta-sigmoid` | CUDA (delta-net) | 起動 −4,080 発 |
| 24 | `fuse-gdn-state-gather` | CUDA (delta-net) | decode の 1.4% |
| 25 | `gdn-gather-single-snapshot` | CUDA (delta-net) | コピー −7.3 ms |
| 26 | `cpy-fused-rows` | CUDA | カーネル 1.9 倍 |
| 27 | `fuse-concat-gather` | CUDA (delta-net) | +0.74% |
| 28 | `top-k-partial` | CUDA | カーネル 7.6 倍、decode の約 1.1% |

適用範囲タグの定義・詳細・停止スイッチ・棄却した代案は
**[docs/patches.ja.md](docs/patches.ja.md)** を参照。

## 使い方

### Nix

```nix
{
  inputs.llama-cpp-p100-patches.url = "github:shinbunbun/llama-cpp-p100-patches";
}
```

順序付きのパッチ列を自分のビルドに適用するか:

```nix
llama-cpp-patched = pkgs.llama-cpp.overrideAttrs (old: {
  patches = (old.patches or [ ]) ++ inputs.llama-cpp-p100-patches.lib.patches;
});
```

overlay を使う。ただし**最後に**適用し、`b10133` の無改変ツリーを前提とすること。
fuzz 0 のパッチは、同じ行を先に書き換えたものがあると reject する:

```nix
nixpkgs.overlays = [ inputs.llama-cpp-p100-patches.overlays.default ];
```

`#llama-cpp-sm60` は Pascal 向けに直接ビルドする。バイナリキャッシュは無いので
CUDA のフルビルドになる (分ではなく時間単位)。

`nixpkgs` の `cudaCapabilities` は既定が 7.5 以上なので、そのままでは生成物に
sm_60 コードが無く、実行時に *"named symbol not found"* で落ちる。Pascal 向けには
`cudaCapabilities = [ "6.0" ]` と **CUDA 12.x** が要る (CUDA 13 は Pascal の
コード生成を削除しているので、flake 側で `cudaPackages_12` を明示的に固定している)。

### Nix 以外

```console
$ git clone --branch b10133 https://github.com/ggml-org/llama.cpp
$ cd llama.cpp
$ for p in ../llama-cpp-p100-patches/patches/*.patch; do
    patch -p1 -F0 < "$p" || { echo "FAILED: $p"; break; }
  done
```

番号順に適用し、**`-F0` を外さないこと**。ここで危険なのは reject ではなく
**誤適用**である。fuzz を許すと、適用に成功したまま hunk が別の場所に入る。
最初の失敗で止めること — 半端に当たったツリーの上に続けて当ててはいけない。

全部入れる必要はないが、自由に選べるわけでもない。複数のパッチが同じ行に触り、
後のパッチが前のパッチの上に成り立っている。途中の 1 本を抜くと、通常は残りの
リベースが必要になる。

### 新しい llama.cpp へのリベース

1. `nix/patches.nix` の `llamaCppVersion` **と** `flake.nix` の `llama-cpp-src`
   入力を上げる。両者は別のリテラルなので必ず同時に変えること
2. `nix flake check` を実行する。当たらなくなったパッチだけが落ちる
3. 落ちた各パッチについて判断する。**upstream が直した** — パッチを削除し、
   README の行と `docs/patches.md` の項目も消して、その旨を書く。
   **移動しただけ** — 新しいツリーに対して再生成する
4. `test-backend-ops` を回し、測り直す。**まだ当たることと、まだ価値があることは別**
   である。ここにあるパッチのいくつかは upstream が別のハードウェア向けに調整した値が
   理由で存在しており、upstream がその値を見直している可能性がある

## 測定環境

- Tesla P100-PCIE-16GB (GP100, sm_60)、CUDA 12.9、ドライバ 580.x
- Qwen3.5 9B dense と 256 expert の Qwen3.5-MoE (アクティブ 8、41 層)、
  Q4_1 / Q5_1 / IQ 系の重み
- MTP による投機デコード。したがってホットパスは幅 5 の検証バッチ
- 固定 3 プロンプトに対する `llama-server` の end-to-end スループット。サンプラーは
  現実的な設定 (temperature 0.7 / top-p 0.8 / top-k 20) — ここではサンプラー設定が
  結論を数ポイント動かす (測定手法の項を参照)
- 出力が壊れる変更では固定プロンプトが比較不能になるので `llama-batched-bench -npl 5`
- 後半の測定は 2 プロンプトの幾何平均である。3 本のうち 1 本が二峰性で、分散の約 9 割を
  出していたことが分かったため。「3 プロンプト」と書いてある項目はそれ以前の測定である

これに必要だった測定の作法 — 熱の制御、ノイズ下限より小さい効果の測り方、
隔離マイクロベンチが 2 回続けて外した話、融合の罠 — は
**[docs/benchmarking.ja.md](docs/benchmarking.ja.md)** にまとめてある。
おそらくパッチ本体よりも応用が利く。

## 注意

- **decode 向け**、特に投機デコードの検証バッチ幅 2〜5 の単一ストリーム decode に
  合わせてある。大バッチのサービングは対象外
- `sm_60` / `pre-Volta` / `pre-Turing` のパッチは、upstream が新しいハードウェア向けに
  調整した値を変更している。`pre-Turing` には Volta が含まれるが未実測である。
  07 はそもそも分岐していない。Turing 以降で使う前に必ず測ること
- `CUDA (delta-net)` のパッチは gated delta-net を持つモデル (Qwen3-Next / Qwen3.5)
  でしか発火しない。それ以外では発火しないだけで、コストはグラフごとのパターン照合のみ
- `nix flake check` が保証するのは「パッチが当たること」だけで、llama.cpp の
  テストは走らせていない

## コントリビュート

Issue / PR 歓迎。最も価値があるのは**手元に無いハードウェアでの実測**である —
P40 / GTX 10xx / Maxwell、あるいは Turing 以降。`sm_60` 以外の適用範囲タグは
すべて、1 枚の GP100 からの推論でしかない。

## ライセンス

llama.cpp に合わせて MIT。[LICENSE](LICENSE) を参照。
本リポジトリは非公式であり、llama.cpp プロジェクトとは無関係である。
