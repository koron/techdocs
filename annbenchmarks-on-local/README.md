# ANN-Benchmarks をローカルで動かす

## 準備

まずは[ドキュメント通り](https://github.com/erikbern/ann-benchmarks/?tab=readme-ov-file#install)に

1. Clone the repo.
2. Run `pip install -r requirements.txt`
3. Run `python install.py` to build all the libraries inside Docker containers (this can take a while, like 10-30 minutes).

Step. 3は `--algorithm` オプションを使ってHNSWを実装している以下のアルゴリズムに限定する。 `(+)` 付きは実行時に注意が必要。

* faiss_hnsw
* hnswlib
* nmslib (+)
* pgvector
* vespa (+)
* voyager

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
