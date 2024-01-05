# llama.cppの量子化の話

llama.cpp のモデルファイル `.gguf` には
量子化方法の情報をファイル名に入れ込む習慣がある。

例 `q4_0` `q4_1` `q4_K_M` `q4_K_S`

`q4` の `4` はビット数(BPW: bits per weight)であることがわかってる。
`M` や `S` はサイズであることがわかってる。 ([参考](./README.md))

`0`, `1`, `K` がなんなのかわからないので調べた。

これらのパラメータは全部の組み合わせが可能なわけではない。
有効な組み合わせに対してquantizerとdequantizerが関数として定義されている。
結果、利用可能な量子化手法(パラメーター)は限られている。
([参考](https://github.com/ggerganov/llama.cpp/blob/b3a7c20b5c035250257d2b62851c379b159c899a/ggml-quants.h#L170-L182))

## `q4_0` 量子化

1. 32個のfloatを1ブロックとし量子化する
2. ブロック中の絶対値が最大の値を探し出し `amax` とする
3. `[-amax, amax]` の区間を16個(=2^4)の均等なサブ区間へ分ける
4. ブロック全部をサブ区間へ変換=量子化する

### 感想

最悪のケースで量子化領域の半分しか使えない。
全部のfloatが正だった場合に負のサブ区間は決して使われないことになる。
っていうか、そんなことわかり切ってるのになんでこれを用意したのか、
いまだに使われてるのかがわからない。
速いから?

### 参考リンク:

* <https://github.com/ggerganov/llama.cpp/blob/b3a7c20b5c035250257d2b62851c379b159c899a/ggml-quants.h#L10-L15>
* <https://github.com/ggerganov/llama.cpp/blob/b3a7c20b5c035250257d2b62851c379b159c899a/ggml-quants.c#L444>

## `q4_1` 量子化

1. 32個のfloatを1ブロックとし量子化する
2. ブロック中の最大・最小の値を探し出し `max`, `min` とする
3. `[min, max]` の区間を16個(=2^4)の均等なサブ区間へ分ける
4. ブロック全部をサブ区間へ変換=量子化する

### 感想

シンプルかつ `q4_0` に比べて表現力に無駄がない。
値分布が偏ってるときに無駄になるがそれはまた別に考えるんだろう。
主に `K` がそれじゃないかと推測できる。

### 参考リンク:

* <https://github.com/ggerganov/llama.cpp/blob/b3a7c20b5c035250257d2b62851c379b159c899a/ggml-quants.h#L17-L23>
* <https://github.com/ggerganov/llama.cpp/blob/b3a7c20b5c035250257d2b62851c379b159c899a/ggml-quants.c#L485>

## `q4_K` 量子化

(WIP: 調べてる最中)

### 解読メモ

* `QK_K` - ブロックサイズ。64 or 256 でコンパイル時に決まる。
* `nb` - ブロック数
* `x` は1ブロック処理するごとに更新される `x += QK_K`
* サブブロック - 1ブロックを32個ずつに分割する。なので 2 or 4 サブブロックになる。
    以下はサブブロックごとの手続き
    * $`{avX} = \frac{\sqrt{\displaystyle\sum_{i=0}^{31} x_{b+i} ^ 2}}{32}`$
    * $`w_{i} = {avX} + |x_{i}|`$
    * サブブロックごとに `scale` と `min` を計算 (`make_qkx2_quants()`) TODO!
* サブブロックを越えて `scale` の最大値(`max_scale`)と `min` の最大値(`max_min`)を計算
* 量子化
    * TODO: ブロックサイズが64か256かで実装が別れてるところ
    * TODO: 共通のところ
*   1ブロックあたりに保存しているもの

    `QK_K` が 64 か 256 かで細かい違いはあるけども、本質的に帆損しているものは一緒。

    * `d` float16
    * `dmin` float16
    * scales & mins - 各サブブロック数分 `QKK_256` では6ビット、 `QKK_64` では 4 ビット
    * quants - `QK_K/2` バイト。4ビット量子化なので。

### 参考リンク:

* <https://github.com/ggerganov/llama.cpp/blob/b3a7c20b5c035250257d2b62851c379b159c899a/ggml-quants.h#L104-L123>
* <https://github.com/ggerganov/llama.cpp/blob/b3a7c20b5c035250257d2b62851c379b159c899a/ggml-quants.c#L1826>
