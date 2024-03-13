# ANN Benchmarksとはなんじゃらほい?

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

## 具体的な計測方法を確認

algorithms/faiss をチェック。
BaseANNクラスを派生して、各アルゴリズムを作る。
`query`, `batch_query` で検索する。Faissではこのあたりは共通化される。
`batch_query` の結果は `get_batch_results` で取得する。非同期になるとかありそうだが、詳細は不明。
`fit`でインデックスを、中味を含めて作ってる。
`set_query_arguments` で固有のパラメータを設定しているようだが、引数が自由なので使い方・制約がわからない。
`get_additional` は dict 相当を返してる。アルゴリズムごとの追加情報だとは思うが、使われ方が不明。

`FaissIVFPQfs` のインデックスの作り方。
`index_factory` で `IVF{n}PQ{d}x4fs` なインデックスを作り、train & add して `base_index` とする。
それを `base_index` から `IndexRefineFlat` で新たな `refine_index` を作る。
`set_query_arguments` の `k_reorder` パラメーターが0なら `base_index` を、0以外なら `refine_index` を使う。
この違いがどのような意図なのかは、現時点では不明。Faissのほうを調べる必要あり。

こんなのが見つかった。
[Faiss解説シリーズ（第一回）](https://crumbjp.hateblo.jp/entry/2021/05/05/Faiss%E8%A7%A3%E8%AA%AC%E3%82%B7%E3%83%AA%E3%83%BC%E3%82%BA%EF%BC%88%E7%AC%AC%E4%B8%80%E5%9B%9E%EF%BC%8I9)
Faissのインデックスの概要解説として良い。
この記事によるとRefineは、検索結果のデータ距離を正確算出にするために、元ベクトルを保存しておく1方式のこと。

次の記事は、固有のパラメーターについて示唆がある。
[Faiss解説シリーズ（第二回）ハマりポイント集](https://crumbjp.hateblo.jp/entry/2021/05/06/Faiss%E8%A7%A3%E8%AA%AC%E3%82%B7%E3%83%AA%E3%83%BC%E3%82%BA%EF%BC%88%E7%AC%AC%E4%BA%8C%E5%9B%9E%EF%BC%89%E3%83%8F%E3%83%9E%E3%82%8A%E3%83%9D%E3%82%A4%E3%83%B3%E3%83%88%E9%9B%86)

こちらの記事は、アルゴリズムの組み合わせを変えて、そのパフォーマンス特性を検証している。
Faissの強みがよくわかる。
[Faiss解説シリーズ（第三回）パフォーマンス](https://crumbjp.hateblo.jp/entry/2021/05/07/Faiss%E8%A7%A3%E8%AA%AC%E3%82%B7%E3%83%AA%E3%83%BC%E3%82%BA%EF%BC%88%E7%AC%AC%E4%B8%89%E5%9B%9E%EF%BC%89%E3%83%91%E3%83%95%E3%82%A9%E3%83%BC%E3%83%9E%E3%83%B3%E3%82%B9)

閑話休題

config.ymlにベンチマークのパラメタライズ等が記載されてる。
これを読む人がいるはず。
このあたりに繋がってた。
* <https://github.com/erikbern/ann-benchmarks/blob/1bf6cfd15ab410c2059ab277c1ee1c1fe85f51fe/run.py#L7>
* <https://github.com/erikbern/ann-benchmarks/blob/1bf6cfd15ab410c2059ab277c1ee1c1fe85f51fe/ann_benchmarks/main.py#L311>

scannの内容を確認。
<https://pypi.org/project/scann/> を使ってる。
[`query_args` のペアが多い。](https://github.com/erikbern/ann-benchmarks/blob/1bf6cfd15ab410c2059ab277c1ee1c1fe85f51fe/ann_benchmarks/algorithms/scann/config.yml#L12-L15)
これがプロットが多い理由だろうか?
`args` は `__init__` に引き渡されるらしい。
Scannでは `n_leaves`, `avg_threshold`, `dims_per_block`, `dist` の4つ。
見た目、あってる。
`query_args` のほうは `leaves_to_search`, `reorder` のペアらしい。

argorithms以下は、単なるAdapterとして機能する。

ベンチマークのプロシージャー:

1. [`load_and_transform_dataset`](https://github.com/erikbern/ann-benchmarks/blob/1bf6cfd15ab410c2059ab277c1ee1c1fe85f51fe/ann_benchmarks/runner.py#L138)

    データセットをダウンロードし、読み込み。
    次元数を決め、train、test、distance(たぶん距離関数)

2. [`build_index`](https://github.com/erikbern/ann-benchmarks/blob/1bf6cfd15ab410c2059ab277c1ee1c1fe85f51fe/ann_benchmarks/runner.py#L162)

    アルゴリズムの `fit` でインデックス作成

3. `set_query_arguments`

    とくになし

4. [`run_individual_query`](https://github.com/erikbern/ann-benchmarks/blob/1bf6cfd15ab410c2059ab277c1ee1c1fe85f51fe/ann_benchmarks/runner.py#L22)

    クエリ実行。単体とバッチがある。
    バッチの結果は `List of ({総時間}, List of({ベクトルidx}, {実際の距離}))`
    単体は `List of({ベクトルidx}, {実際の距離})`

5. [`store_results`](https://github.com/erikbern/ann-benchmarks/blob/1bf6cfd15ab410c2059ab277c1ee1c1fe85f51fe/ann_benchmarks/results.py#L41)

    HDF5 形式でそのまま保存するだけ見たい。
    グラフに出してる形への整形は別の場所。

整形はこのあたりか [`get_recall_values`](https://github.com/erikbern/ann-benchmarks/blob/1bf6cfd15ab410c2059ab277c1ee1c1fe85f51fe/ann_benchmarks/plotting/metrics.py#L14)

## グラフ内の点同士の関係を調べる

上記は `runner.run()` のプロシージャである。
この関数はアルゴリズムとデータセットを指定してベンチマークを実行するもの。
インデックスを1度作成後、パラメータに応じて複数回テスト(クエリ)を実行する。
パラメータはconfig.yml内の`query_args`と`query_arg_groups`の組み合わせになる。
が`query_arg_groups`は既に使われてないようだ。

config.ymlの`args` はdefinitions.pyの`prepare_args()`で取り出している。
最終的に`Definition.arguments`に格納している。
definitions.pyの`instantiate_algorithm`でAlgorithmのコンストラクターに渡している。

アルゴリズムとconfig.ymlと各種パラメータの関係はわかった。
アルゴリズムとデータセットを選択すると、ベンチマーク結果全体が得られる。
結果全体には、クエリ引数セットごとに結果すなわち精度と時間が格納されている。
順番つまりプロットしたときの、或るシリーズの各点の前後関係は、クエリ引数セット=`query_args` の要素順序に依存している、と推定される。

最後にそれを確定してしまいたい。

結果は `[ (時間, [(候補単語, 距離)]) ]` の形。
各クエリにかかった時間、出力された候補、そのペアがクエリの数だけ。

他のメタデータとあわせてHDF5に追記する。
この時 `query_arguments` の順番で、その値とともに格納される。

`compute_metrics()` で全ファイルから再構成。
`res` すなわち `args`, `query_arguments` の組み合わせの順番で再構成される。

グラフのプロットの各点は config.yml の `args`, `query_arguments` の組み合わせの順である。
