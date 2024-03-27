# ANN-Benchmarks をローカルで動かす

## 準備

まずは[ドキュメント通り](https://github.com/erikbern/ann-benchmarks/?tab=readme-ov-file#install)に

1. Clone the repo.
2. Run `pip install -r requirements.txt`
3. Run `python install.py` to build all the libraries inside Docker containers (this can take a while, like 10-30 minutes).

Step. 3は `--algorithm` オプションを使ってHNSWを実装している以下のアルゴリズムに限定する。

* `faiss_hnsw` (`hnsw(faiss)`)
* `hnswlib`
* `pgvector`
* `vespa` (`hnsw(vespa)`)
* `voyager`
* `nmslib` (x 実行できず)

実行例:

```console
$ python install.py --algorithm faiss_hnsw

$ python install.py --algorithm hnswlib

...
```

## 実行(計測)

以下の形で Faiss のHNSWだけを計測しようとした。

```console
$ python run.py --algorithm faiss_hnsw
```

バグを2つ修正して実行できた。

1. 空のYAMLファイル(設定用)を読み込むとNoneが返ってくるケースの考慮漏れ
2. `faiss_hnsw` の名前の不整合

    * アルゴリズム名 `hnsw(faiss)`
    * Dockerタグ名 `faiss_hnsw`

<details>
<summary>修正用パッチ</summary>

```
diff --git a/ann_benchmarks/algorithms/faiss_hnsw/config.yml b/ann_benchmarks/algorithms/faiss_hnsw/config.yml
index 84d6baa..9ede091 100644
--- a/ann_benchmarks/algorithms/faiss_hnsw/config.yml
+++ b/ann_benchmarks/algorithms/faiss_hnsw/config.yml
@@ -3,7 +3,7 @@ float:
   - base_args: ['@metric']
     constructor: FaissHNSW
     disabled: false
-    docker_tag: ann-benchmarks-faiss
+    docker_tag: ann-benchmarks-faiss_hnsw
     module: ann_benchmarks.algorithms.faiss_hnsw
     name: hnsw(faiss)
     run_groups:
diff --git a/ann_benchmarks/definitions.py b/ann_benchmarks/definitions.py
index 5720491..e8ad692 100644
--- a/ann_benchmarks/definitions.py
+++ b/ann_benchmarks/definitions.py
@@ -138,6 +138,8 @@ def load_configs(point_type: str, base_dir: str = "ann_benchmarks/algorithms") -
         with open(config_file, 'r') as stream:
             try:
                 config_data = yaml.safe_load(stream)
+                if config_data is None:
+                    continue
                 algorithm_name = os.path.basename(os.path.dirname(config_file))
                 if point_type in config_data:
                     configs[algorithm_name] = config_data[point_type]
```
</details>

正しい実行方法は以下の通り:

```console
$ python run.py --algorithm 'hnsw(faiss)'
```

1ケース目の実行に15分かかった。9ケースあるので…2h15mかかることになる。
さすがにとめて並行させた方が良さそう。

いったん止めて `--parallelism 5` で再実行。

```console
$ python run.py --parallelism 5 --algorithm 'hnsw(faiss)'
```

16コアCPUにおける使用量が45%くらいなので `10` くらいまで上げてもよさそう。
ハイパースレッドを休ませることを考慮して、実コアレベルで考えれば `8` が妥当か?
`run_groups` は HNSW のケースでは9個なので、
以降の実行では `9` としよう。

結局 `5` でも2時間半かかった。Mが大きくなると構築に時間がかかるのが原因か。

次は hnswlib を一発で実行する。

```console
$ python run.py --parallelism 9 --algorithm hnswlib
```

その後、グラフを書かせるには `plot.py` でも良いが
`create_website.py` のほうが見やすくて良さそう。

```console
$ mkdir website
$ python create_website.py --scatter --outputdir website
```

voyager の大きいところは遅くて7200秒でタイムアウトしてしまった。
`--timeout 14400` を指定して再試。

pgvector はテストケースが少ない&めちゃくちゃ遅い。
遅いからテストケースを少なくしたと考えられる。

追加で次元数が多い `--dataset fashion-mnist-784-euclidean` データセットもテストする。
実行コマンド例:

```console
$ python run.py --timeout 14400 --parallelism 9 --dataset fashion-mnist-784-euclidean --algorithm 'hnsw(faiss)'
$ python run.py --timeout 14400 --parallelism 9 --dataset fashion-mnist-784-euclidean --algorithm 'hnswlib'
$ python run.py --timeout 14400 --parallelism 9 --dataset fashion-mnist-784-euclidean --algorithm 'pgvector'
$ python run.py --timeout 14400 --parallelism 9 --dataset fashion-mnist-784-euclidean --algorithm 'hnsw(vespa)'
$ python run.py --timeout 14400 --parallelism 9 --dataset fashion-mnist-784-euclidean --algorithm 'voyager'
```

`create_website.py` で作ったデータを [results-20230314](https://koron.github.io/techdocs/annbenchmarks-on-local/results-20240314/) に置いた。
考察の続きは [../hnsw-libs-comparison](../hnsw-libs-comparison/) にて行う。

## 考察

前述の結果をみて考察する。

* HNSW実装: pgvector
    * 1桁から2桁は速度が遅い
    * メモリではなくファイルであることに要注意
        * メモリに収まらないサイズを扱えるハズ
        * 純粋に大きな数での速度劣化は?
        * 理論上は `O(log(n))` だと考えられるが、本当にそうか?
        * pgvector 自身にそれを検証するベンチマークがありそう
* HNSW実装: ほか
    * 大きな差異はない: あえて差を見出すならば…
        * 高いRecallの条件で安定して速いのがFaiss
            * 実用観点で使いやすそう
            * 他のアルゴリズムとの組み合わせもしやすい
    * Javaで使える voyager もJava環境という観点で利用しやすい
* グラフの見方
    * データセットを固定してHNSWの実装を比較する場合
        * Recall(横軸)を固定して、実装毎の縦軸の値を比較する。逆ではない
        * アルゴリズムはHNSWであるため、Recallを比較することにあまり意味はない
        * 実装ごとに速度の差が出る
        * インデックスサイズについても大きな差はできない
            * データをどう持つか的な意味で少しは差がある
    * HNSWの実装を固定してデータセットを比較する場合
        * QPS(縦軸)を固定してRecall(横軸)をデータセットで比較する
        * HNSWでクラスタリングしやすいデータセットがわかる
    * サイズの比較はインデックスのみで、データ本体のベクトル圧縮等は考慮されていない
* MNISTデータセット
    * HNSWアルゴリズムにおいて好成績になりやすい
    * MNISTはもともと、明らかに少ない数にクラスタリングされているものを再現するタスク
        * オリジナルMNISTは手書きの数字画像、10種のクラスタリング
        * fashion-mnistはファッションアイテム画像、60種のクラスタリング
    * ローカルリンクが少ない≒QPSが高くなる条件でもRecallが良くなるので、グラフが右側に偏って、実装による差が小さく見える
* GloVeデータセットは単語のベクトル表現で数も多いため、MNISTにくらべると成績がよくない
    * ローカルリンクが少ない条件で、目に見えてRecallが悪くなり、実装による細かい差が大きく見える

### pgvectorの公式なベンチマーク

<https://aws.amazon.com/jp/blogs/database/accelerate-hnsw-indexing-and-searching-with-pgvector-on-amazon-aurora-postgresql-compatible-edition-and-amazon-rds-for-postgresql/>

ベンチマークのデータ件数への依存性ではなく、並列数への依存性、インスタンスタイプによる依存性、pgvectorのバージョンによる依存性を計測したAWSのポスト

データ件数と検索にかかる時間もしくはQPSに関する公式ベンチマークは見つけられなかった。

その過程で次の記事を見つけた。なんだこれは?
[10x Faster than Meta's FAISS](https://www.unum.cloud/blog/2023-11-07-scaling-vector-search-with-intel)

## まとめ

多少手直しが必要であったが、dockerを用いてlocalで ANN Benchmarks を実行し結果を可視化できた。

動かすためのPR2つを出した。

* [#496 fix docker tag for Faiss HNSW](https://github.com/erikbern/ann-benchmarks/pull/496)
* [#497 skip `None` for loaded config data](https://github.com/erikbern/ann-benchmarks/pull/497)

比較結果からHNSW実装とデータセットの関係性を考察した。
