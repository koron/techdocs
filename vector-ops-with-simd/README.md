# ベクトル操作(検索)におけるSIMD利用実態についての調査

[Usearch](https://github.com/unum-cloud/usearch)がベクトル検索においてSIMDを利用することでパフォーマンスを稼いでることが判明した。
そこから現状のベクトル操作(主に検索)におけるSIMDの利用実態はどうなっているのか、という疑問が湧いたので調べてみる。

## 調査メモ1

まずはド定番のnumpyだろう。

2021年のこんな記事が見つかった。

[NumPy 1.20が実行時SIMDサポートと型アノテーションを加えリリースされた](https://www.infoq.com/jp/news/2021/05/numpy-120-typed-SIMD/)

numpy自身のリファレンスマニュアル: [CPU/SIMD Optimization](https://numpy.org/doc/stable/reference/simd/index.html) がある。
universal intrinsicsを使って書いて、ビルド時に有効化していれば、ランタイムで利用可能だよ、と書いてある。
細かい仕様の調査は割愛。

[Universal Intrinsics](https://docs.opencv.org/4.x/df/d91/group__core__hal__intrin.html) は移植性のあるSIMDプログラムを書くための型及び関数の取り決め、ということらしい。

次にPandasで、これにはRustで実装された[polars](https://github.com/pola-rs/polars)がSIMDを使ったもの代替品として提案されている。

次に[Faiss](https://github.com/facebookresearch/faiss)。
もともとfaiss-gpu等もありSIMDも使ってるだろうとは考えていたが、やはりありそう。
ブログ: [4-bit PQの解説 (2024-06-21 松井勇佑先生だ!)] によれば、4-bit PQ のARM  SIMDを使った版を実装してFaissにコントリビュートしたという話。
いわゆる、ただのベクトル距離計算に用いたわけではない点に留意が必要。
なお、ざっくり16並列になるらしい。

Faiss自身には [simdlib.h](https://github.com/facebookresearch/faiss/blob/5893ab77daee3c84ecc74a2c84c18d7cd486fcea/faiss/utils/simdlib.h) というヘッダーライブラリがSIMDの機能を抽象化して提供しており、AVX2, AVX512 および NEON (ARM) の実装があるのが見て取れる。
ここから逆にたどれば、Faissのどの機能がSIMDに対応しているかは読めるだろう。

次に pgvector 。PostgreSQLの拡張である pgvector にはRustで書かれたおおよそ互換性のありそうな代替品: [pgvecto.rs](https://docs.pgvecto.rs/getting-started/overview.html) があった。
こちらはSIMD対応が動的に選択でき速いとのこと。pgvectorはコンパイル時に決まるから柔軟性にかけるよね、という主張にみえる。
あと最大65536次元と次元数が大きい。
比較表は [こちらのブログの末尾](https://blog.pgvecto.rs/pgvector-vs-pgvectors-in-2024-a-comprehensive-comparison-for-vector-search-in-postgresql#heading-summary) で完結にまとめられている。

本家 pgvector では約2週間前(2024-04-08)に距離の計算にSIMDが導入されたようだ。
とてもホットなテーマ。

* L2距離: <https://github.com/pgvector/pgvector/commit/925aa4e048f029491851bf25375890b3ace4a75b>
* 内積: <https://github.com/pgvector/pgvector/commit/9ed39cee67fa79dad78cee2b6f6a1f4c8d8d71b4>
* コサイン距離: <https://github.com/pgvector/pgvector/commit/9ed39cee67fa79dad78cee2b6f6a1f4c8d8d71b4>

### ここまでの感想

* SIMDを応用するのは、計算ライブラリの有名どころでは当然やってた
    * numpyやfaissやpandas alternativeが該当
    * 一方でストレージと絡めたところは、今がホットっぽい
* Rustで実装されてるのが2つもあるのが印象的
* pgvecto.rs は気になる: ann-benchmark ベースの比較をみてみたい
* pgvector では追加されたのが思いのほか最近だった
    * usearchやpgvecto.rsから刺激を受けたか?
    * いずれ managed serviceでも利用できるのだろう
* こうなると usearch の優位性もさほどないのでは?
