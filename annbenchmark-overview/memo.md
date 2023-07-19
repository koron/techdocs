# ANN Benchmarkとはなんじゃらほい?

https://ann-benchmarks.com/

近似近傍計算のベンチマークとその結果。

## 概要

Recall = 真の近傍に一致した割合、全クエリの平均: 1に近づくほど良い。
QPSとRecallをプロットする: 品質の良い近似ほどクエリに時間がかかる。

まず距離関数ごとにまとまってる。
次にアルゴリズムごとにまとまってる。
(HNSWとの組み合わせが気になってる)

## 距離関数のの一覧

* [角距離](https://en.wikipedia.org/wiki/Angular_distance) 内積とほぼ一緒だろう
* ユークリッド距離: 普通に想像する距離。二乗して足して平方根を取るやつ
* [ハミング距離](https://ja.wikipedia.org/wiki/%E3%83%8F%E3%83%9F%E3%83%B3%E3%82%B0%E8%B7%9D%E9%9B%A2): ビット表現における、異なるビットの数
* [ジャッカード係数](https://en.wikipedia.org/wiki/Jaccard_index): {共通する要素数} / {重複しない全要素数}

## アルゴリズム一覧

気になるアルゴリズムはこのあたり。

* faiss-ivf
* scann
* hnswlib
* hnsw(faiss)
* faiss-ivfpqfs
* redisearch
* luceneknn

<details>
<summary>全アルゴリズム</summary>

* faiss-ivf
* scann
* pgvector
* annoy
* glass
* hnswlib
* BallTree(nmslib)
* vald(NGT-anng)
* hnsw(faiss)
* NGT-qg
* qdrant
* n2
* Milvus(Knowhere)
* qsgngt
* faiss-ivfpqfs
* mrpt
* redisearch
* SW-graph(nmslib)
* NGT-panng
* pynndescent
* vearch
* hnsw(vespa)
* vamana(diskann)
* flann
* luceneknn
* weaviate
* puffinn
* hnsw(nmslib)
* bruteforce-blas
* tinyknn
* NGT-onng
* elastiknn-l2lsh
* sptag
* ckdtree
* kd
* opensearchknn
* datasketch
* bf

</details>

## データセット

推測されるデータセットの名称フォーマット:

    {ソース名}-{次元数}-{距離関数}

### ソース名

情報元: <https://github.com/erikbern/ann-benchmarks/blob/main/README.md#data-sets>

* DEEP1B <http://sites.skoltech.ru/compvision/noimi/>
* Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>
* GIST <http://corpus-texmex.irisa.fr/>
* GloVe <https://nlp.stanford.edu/projects/glove/>
* Kosarak <http://fimi.uantwerpen.be/data/>
* MNIST <http://yann.lecun.com/exdb/mnist/>
* MovieLens-10M <https://grouplens.org/datasets/movielens/10m/>
* NYTimes <https://archive.ics.uci.edu/dataset/164/bag+of+words>
* SIFT <http://corpus-texmex.irisa.fr/>
* Last.fm <https://github.com/erikbern/ann-benchmarks/pull/91>

いずれも[HDF5フォーマット](https://ja.wikipedia.org/wiki/Hierarchical_Data_Format#HDF5)で提供されている

次元数とTrain sizeと角度がポイント。近傍数は一律100。

次元はKosarakとMovieLens-10Mが1万超で大きい。
次点でGIST, Fashion-MNIST, MNIST が1000弱。

# グラフを読んでみる

よくわかってない指標

*   Build times: Recall/Build timesより
*   Distance computations: 計算回数ってこと?
*   Relative Error: 定義が不明
*   Candidates generated: 生成された候補(数)、という訳ではあろうが、それがどういう意味を持つのか?
*   Epsilon {0.01, 0.1}: 誤差0.01もしくは0.1のRecallと読める。

    0.01よりも0.1のほうが同RecallにおけるQPSが大きいので、ほぼあってる。
    しかしどう実現・統計処理しているのかがわからない。

ただ各グラフにはどちらの方向が優位なのか書いてあるので、それだけ意識すればよいともいえる。

## アルゴリズムによる比較

### hnswlib vs hnsw(faiss)

hnswlibのほうが全般的に速い。
Recallが1.0近辺になると同じくらいの速度になる。

Index sizeはとんとん。hnsw(faiss)が気持ち勝る(小さい)か?
hnswlibはRecallが上がらない傾向にある。

Relative Errorはhnsw(faiss)が勝りそう。
よりエラーが少ない。

大きな差はないので、シンプルなhnswlibでも良いかなという気にはなる。

### hnsw(faiss) vs faiss-ivf

hnsw(faise)のほうがindexが小さいようだ。
faiss-ivfはエラーが発生しにくい。そりゃ当然なんだけども。
hnsw(faise)はエラーがある代わりに速い。

### faiss-ivf vs faiss-ivfpqfs

faiss-ivfpqfsのほうがちょっと速い。
距離計算の回数が減ってる。
Indexサイズが減ってる。
エラーは少し良いか?

### faiss-ivfpqfs vs scann

scannのほうが…
低品質時に速い。
他をみてもなんとなくscannのほうが、精度がよく速そうだ。
faissのfastscanはscannにも勝るとも劣らないとの主張の根拠はどこだ?
