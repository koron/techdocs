# Usearch

<https://github.com/unum-cloud/usearch>

気になる売り文句:

* FAISSよりHNSWが10倍速い
* C++ 11 によるヘッダーライブラリ
* SIMDで速い
* ファイルからの直アクセス

## ANN Benchmarksで試してみた

[annbenchmarks-on-local](../annbenchmarks-on-local) からの引用

最新の ann-benchmarks に合わせて [オリジナルPR](https://github.com/erikbern/ann-benchmarks/pull/451) を修正して計測。
<https://github.com/koron/ann-benchmarks/tree/usearch-engine-update/ann_benchmarks/algorithms/usearch>

```console
# Docker image 作成
$ python install.py --algorithm usearch

# 計測
$ python run.py --timeout 14400 --parallelism 9 --dataset fashion-mnist-784-euclidean --algorithm usearch

# グラフ出力
$ python create_website.py --scatter --outputdir website
```

f32を試した。
というよりf8はSIMDが利用できず試せなかった。

プリコンパイルのPythonライブラリがSIMDを無効化してビルドされているようだ。
自分でコンパイルするのも失敗した。
コンパイルの失敗理由を調査するまではしなかった。
AVX512を利用できないCPUを使っており(というか使えるCPUのほうがが少ない)、
仮にコンパイルできてもSIMDの性能を発揮しきれない可能性が高かった。

クエリはpgvectorよりは速いが、Faissよりは遅い。
彼らの主張ではもっとベクトル数が大きい時に速くなるとのことなので、そんなものだろう。

インデックスは、pgvectorよりも作成に時間がかかり、かつ大きくなった。
使いどころが極めて難しそう。

SIMD(AVX512)でf8使って100M ベクター超えとなると、
ちょっとann-benchmarksどころか自前で確認するのも大変そう。

## ソース探検

本体はココ: <https://github.com/unum-cloud/usearch/tree/main/include/usearch> の3つのヘッダーファイル。

言語バインディングごとに、このヘッダーを読み込んで、言語毎に足りない・必要な部分を実装する。
例: Goだと [C向けバインディング](https://github.com/unum-cloud/usearch/tree/main/c) の上に [Goのバインディング](https://github.com/unum-cloud/usearch/tree/main/golang) を重ねてる。

`USEARCH_USE_SIMSIMD` を定義することで [SIMSIMD](https://github.com/ashvardanian/simsimd) を通じてSIMDを利用している。
SIMSIMDのpre-requirementも参照する必要がある。

SIMSIMDもヘッダーライブラリっぽい。Python, Rust, Goのバインディングが確認できた。

### index.hpp

* include バリア
* バージョン表明
* 環境判定
* include: C++ STL
* マクロ
    * alignment
    * debug
* ユーティリティ関数・テンプレート・クラス

`index_gt` が目的の手続きを実装している。

その構成クラス群:

```
$ sed -ne '1689,3610p' < include/usearch/index.hpp  | grep '^\s*\(struct\|class\)\b'
class index_gt {
    struct precomputed_constants_t {
    struct candidate_t {
    class node_t {
    class neighbors_ref_t {
    struct usearch_align_m context_t {
    struct copy_result_t {
    struct add_result_t {
    struct match_t {
    class search_result_t {
    struct cluster_result_t {
    struct stats_t {
        struct slot_level_t {
    struct node_lock_t {
    struct candidates_range_t;
    class candidates_iterator_t {
    struct candidates_range_t {
```

## まとめ

ざっくり速度に有利そうなのは3点

1. SIMDによる距離関数の高速化
2. SIMDに特化した(実質)量子化
3. マルチスレッド aware なコード

    * ロック
    * 最大スレッド数に応じたバッファ
