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

## `q4_0` 量子化

1. 32個のfloatを1ブロックとし量子化する
2. ブロック中の絶対値が最大の値を探し出し `amax` とする
3. `[-amax, amax]` の区間を16個(=2^4)にサブ区間へ分ける
4. ブロック全部をサブ区間へ変換=量子化する

参考リンク:

* <https://github.com/ggerganov/llama.cpp/blob/b3a7c20b5c035250257d2b62851c379b159c899a/ggml-quants.h#L10-L15>
* <https://github.com/ggerganov/llama.cpp/blob/b3a7c20b5c035250257d2b62851c379b159c899a/ggml-quants.c#L444>

## `q4_1` 量子化

1. 32個のfloatを1ブロックとし量子化する
2. ブロック中の最大・最小の値を探し出し `max`, `min` とする
3. `[min, max]` の区間を16個(=2^4)にサブ区間へ分ける
4. ブロック全部をサブ区間へ変換=量子化する

参考リンク:

* <https://github.com/ggerganov/llama.cpp/blob/b3a7c20b5c035250257d2b62851c379b159c899a/ggml-quants.h#L17-L23>
* <https://github.com/ggerganov/llama.cpp/blob/b3a7c20b5c035250257d2b62851c379b159c899a/ggml-quants.c#L485>

## `q4_K` 量子化

(WIP: 調べてる最中)

参考リンク:

* <https://github.com/ggerganov/llama.cpp/blob/b3a7c20b5c035250257d2b62851c379b159c899a/ggml-quants.h#L104-L123>
* <https://github.com/ggerganov/llama.cpp/blob/b3a7c20b5c035250257d2b62851c379b159c899a/ggml-quants.c#L1826>
