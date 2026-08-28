# llama-cpp-p100-patches

[llama.cpp](https://github.com/ggml-org/llama.cpp) の性能パッチ 30 本。
**Tesla P100 (GP100, sm_60)** 上で開発し、実測した。

English: [README.md](README.md)

```console
$ nix build github:shinbunbun/llama-cpp-p100-patches#llama-cpp-sm60
```

出発点は GP100 である。Pascal の中でこのチップだけが DP4A を持たず、代わりに
フルレートの HFMA2 を持つ。そのため llama.cpp の量子化 matmul はエミュレーションと
cuBLAS に落ちる一方で、このカードが本当に得意な命令は使われないままになる。
名前に `sm60` が付くパッチは、この非対称性を突いたものである。

**とはいえ、30 本のうち Pascal 世代に限定されているのは 7 本だけである。** もう 1 本は
アーキテクチャで分岐しない定数を変えるので、全 GPU に影響する。残る 22 本はハードウェアに
依存しない。内訳はカーネル融合、添字計算の修正、ホスト側のサンプラー経路 2 本、
スケジューラ 2 本、語彙全体のソートを避ける `top_k`、そして参照テーブル 2 種の共有メモリ化で
ある。ただしこのうち 5 本は gated delta-net を持つモデルでしか発火せず、1 本は特定の
モデル機能を必要とする。
下の表の各行に適用範囲のタグを付けてあり、[docs/patches.ja.md](docs/patches.ja.md) は
その分類で章立てしてある。

どのパッチにも実測の裏づけがある。根拠は**試したうえで棄却した代案も含めて**、
パッチがソースに追加するコメントに書いてある。

## 全体でどれだけ速くなるか

同じカード・同じモデル・同じサーバオプションで、無改変の `b10133` と比較した値である。

| モデル | 無改変 | パッチ適用 | |
|---|---:|---:|---:|
| Qwen3.5 9B dense (Q4_1、MTP ヘッド Q4_1) | 76.99 t/s | **136.11 t/s** | **+76.8%** |
| Qwen3.6 35B-A3B MoE (Q2_K_XL、dense 層 Q5_1) | 67.01 t/s | **120.57 t/s** | **+79.9%** |

`llama-server` の end-to-end decode スループットで、3 プロンプトの幾何平均。MTP による
投機デコード (n-max 4、p-min 0.75) と現実的なサンプラー (temperature 0.7 / top-p 0.8 /
top-k 20) を使っている。両腕を交互に測り、後半は順序を反転、各測定の前に GPU を 58 °C
以下まで冷ましてある。dense は 6 ラウンド (SD 無改変 0.19% / パッチ適用 0.08%)、
MoE は 4 ラウンド (0.16% / 0.12%)。

**どちらもパッチ 29 / 30 より前の測定である。** dense のほうは Q4_1/Q5_1 だけなので
影響を受けない。MoE のほうはバイトの 31.7% が IQ3_XXS、47.9% が IQ2_XS で、別々の
2 周 A/B でパッチ 29 から 1% ほど、パッチ 30 から 1.5% 伸びるので、上の値はわずかに
過小である。

これは**パッチ全体を無改変と比べた値**であり、下の表のパッチ単体の値を足したものでは
ない。単体の値はそれぞれ、その時点のスタックに対して測ったもので、合成できない。

## 状態

- llama.cpp **`v0.2.0`** に対して生成しており、**fuzz 0・オフセット 0** で適用できる
  (`nix flake check` がこの両方と、`nix/patches.nix` の順序付きリストが `patches/` の
  中身と一致することを検証する)
- upstream には未提出。明快な候補が 3 本ある。11 `penalties-direct` /
  21 `sched-reset-lazy` / 28 `top-k-partial` はいずれもアーキテクチャ非依存で、
  出力はビット一致、扱えない入力では元の経路にフォールバックする。出していないのは
  単に手が回っていないからにすぎない
- 全パッチを当てたまま P100 で **`test-backend-ops` が通る**。`v0.2.0` で 13,352 件・
  失敗 0 で、同じ機械の無パッチ `v0.2.0` 版と件数も結果も一致する
- パッチ自身が断っていない限り、出力は無パッチ版と**ビット一致**する。出力が変わるのは
  3 本だけで (03 は 1 行を超える場合、12 は通る幅で、15 は設計上)、それぞれ根拠を添えて
  明示してある
- **このリポジトリは縮んでいくことを前提にしている。** upstream が直したものは
  ここから消すべきであり、価値はコードだけでなく測定値の側にもある

## 内容

パーセントは下記の環境における end-to-end スループットである。µs・起動回数・倍率を
書いてあるセルは**カーネル単体の値**で、それらのパッチは end-to-end ではノイズ下限以下
だった (各項目にその旨を書いてある)。**値は足し合わせられない。** どれも、その時点の
スタックに対して測った値である。

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
| 10 | `mmvq-q8-1-activation-cache` | CUDA | +1.17% / +0.88%、この形では約 0.5% 少ない |
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
| 29 | `mmvq-iq3xxs-grid-smem` | CUDA | decode +3.1〜7.6% (dense)、出力ビット一致 |
| 30 | `mmvq-ksigns-smem` | CUDA | decode +0.2〜1.8%、カーネル IQ3_XXS −9.3%、出力ビット一致 |

適用範囲タグの定義、詳細、停止スイッチ、棄却した代案は
**[docs/patches.ja.md](docs/patches.ja.md)** を参照。

## 使い方

### Nix

```nix
{
  inputs.llama-cpp-p100-patches.url = "github:shinbunbun/llama-cpp-p100-patches";
}
```

順序付きのパッチリストを、そのまま自分のビルドに適用する:

```nix
llama-cpp-patched = pkgs.llama-cpp.overrideAttrs (old: {
  patches = (old.patches or [ ]) ++ inputs.llama-cpp-p100-patches.lib.patches;
});
```

あるいは overlay を使う。ただし**最後に**適用し、`v0.2.0` の無改変ツリーに当てること。
fuzz 0 のパッチは、同じ行を先に書き換えたものがあると reject される:

```nix
nixpkgs.overlays = [ inputs.llama-cpp-p100-patches.overlays.default ];
```

`#llama-cpp-sm60` は Pascal 向けに直接ビルドする。バイナリキャッシュは無いので、
CUDA をソースからフルビルドすることになる (分単位ではなく時間単位)。

`nixpkgs` の `cudaCapabilities` は既定が 7.5 以上なので、そのままではバイナリに
sm_60 のコードが残らず、実行時に *"named symbol not found"* で落ちる。Pascal 向けには
`cudaCapabilities = [ "6.0" ]` と **CUDA 12.x** が必要である (CUDA 13 は Pascal の
コード生成を削除しているため、flake 側で `cudaPackages_12` を明示的に固定している)。

### Nix 以外

```console
$ git clone --branch v0.2.0 https://github.com/ggml-org/llama.cpp
$ cd llama.cpp
$ for p in ../llama-cpp-p100-patches/patches/*.patch; do
    patch -p1 -F0 < "$p" || { echo "FAILED: $p"; break; }
  done
```

番号順に適用し、**`-F0` を外さないこと**。ここで危険なのは reject ではなく**誤適用**の
ほうである。fuzz を許すと、適用は成功したまま hunk が別の場所に入ってしまう。
最初の失敗で止めること。半端に当たったツリーの上に、続けて当ててはいけない。

全部を入れる必要はないが、自由に取捨選択できるわけでもない。複数のパッチが同じ行に触り、
後のパッチが前のパッチの上に成り立っている。途中の 1 本を抜くと、たいていは残りの
リベースが必要になる。

### 新しい llama.cpp へのリベース

1. `nix/patches.nix` の `llamaCppTag` **と** `flake.nix` の `llama-cpp-src`
   入力を上げる。この 2 つは別々のリテラルなので、必ず同時に変えること
2. `nix flake check` を実行する。当たらなくなったパッチだけが落ちる
3. 落ちたパッチごとに判断する。**upstream が直した**のであれば、パッチを削除し、
   README の行と `docs/patches.md` の項目も消して、その旨を書く。**移動しただけ**で
   あれば、新しいツリーに対して再生成する
4. `test-backend-ops` を回し、測り直す。**まだ当たることと、まだ価値があることは別である。**
   ここにあるパッチのいくつかは、upstream が別のハードウェア向けに調整した値が理由で
   存在しており、upstream がその値を見直している可能性がある

## 測定環境

- Tesla P100-PCIE-16GB (GP100, sm_60)、CUDA 12.9、ドライバ 580.x
- Qwen3.5 9B dense と Qwen3.6 35B-A3B (256 expert、アクティブ 8、41 層)。重みは
  Q4_1 / Q5_1 / IQ 系
- MTP による投機デコードを使うので、ホットパスは幅 5 の検証バッチになる
- 固定した 3 プロンプトに対する `llama-server` の end-to-end スループット。サンプラーは
  現実的な設定 (temperature 0.7 / top-p 0.8 / top-k 20)。ここではサンプラーの設定次第で
  結論が数ポイント動く (測定手法の項を参照)
- 出力が変わる変更では固定プロンプトでの比較が成立しないので、
  `llama-batched-bench -npl 5` を使う
- 後半の測定は 2 プロンプトの幾何平均である。3 本のうち 1 本が二峰性で、分散の約 9 割を
  出していたことが分かったためで、「3 プロンプト」と書いてある項目はそれ以前の測定である

これに必要だった測定の作法 (熱の制御、ノイズ下限より小さい効果の測り方、隔離した
マイクロベンチが 2 回続けて外した話、融合の罠) は
**[docs/benchmarking.ja.md](docs/benchmarking.ja.md)** にまとめてある。
おそらくパッチ本体より応用が利く。

## 注意

- **decode 向け**であり、特に投機デコードの検証バッチ幅 2〜5 の単一ストリーム decode に
  合わせてある。大バッチのサービングは対象外である
- `sm_60` / `pre-Volta` / `pre-Turing` のパッチは、upstream が新しいハードウェア向けに
  調整した値を変更している。`pre-Turing` には Volta が含まれるが、Volta は未実測である。
  07 はそもそも分岐していない。Turing 以降で使う前に必ず測ること
- `CUDA (delta-net)` のパッチは、gated delta-net を持つモデル (Qwen3-Next / Qwen3.5)
  でしか発火しない。それ以外のモデルでは発火しないだけで、コストはグラフごとの
  パターン照合だけである
- `nix flake check` が保証するのは、パッチが当たること、nixpkgs が生成時のタグを保って
  いること、そして CUDA 抜きでパッチ適用後のツリーがコンパイルできることの 3 点である。
  CUDA のソースはコンパイルせず、llama.cpp のテストも実測も行わない。これらにはカードが要る

## コントリビュート

Issue / PR を歓迎する。最も価値があるのは**手元に無いハードウェアでの実測**である。
P40 / GTX 10xx / Maxwell、あるいは Turing 以降が該当する。`sm_60` 以外の適用範囲タグは
すべて、1 枚の GP100 からの推論でしかない。

## ライセンス

llama.cpp に合わせて MIT。[LICENSE](LICENSE) を参照。
本リポジトリは非公式であり、llama.cpp プロジェクトとは無関係である。
