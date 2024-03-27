# HNSW ライブラリ比較

おもに知りたいのは速度(構築・検索)速度について。

## 事前検討

HNSWの構築はオンタイムでできる。まとまったトレーニングは不要ということ。
測定は全ベクトルの追加にかかった時間で見るのが吉か。

計測候補: Faiss, pgvector, voyager

少しずつ利用可能なパラメーターが違うのがネックか。
Faissとpgvectorはおおよそフェアな比較が成立する。

pgvector は `--tmpfs` でディスクI/Oのオーバーヘッドを最小にできるはず。
とはいえ通信にかかるオーバーヘッドは消せないか。
起動例:

```console
$ docker run --rm -it -p 5432:5432 -e LANG=C.UTF-8 -e POSTGRES_PASSWORD=abcd1234 --tmpfs /var/lib/postgresql/data pgvector/pgvector:0.6.1-pg16
```

件数を変えて、その際の時間変化を見るべきだろうか。

### 参考資料

* [Faiss におけるHNSW(だけではない)のベンチマーク結果](https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors)

    [コード](https://github.com/facebookresearch/faiss/blob/main/benchs/bench_hnsw.py): [SIFT1M](https://www.tensorflow.org/datasets/catalog/sift1m) を用いてる。
    128次元、100万データ。テストデータセットは1万

*  こういうときこそまず <https://ann-benchmarks.com/> を見ればよいのでは?

    * pgvector: https://ann-benchmarks.com/pgvector.html
    * hnsw(faiss): https://ann-benchmarks.com/hnsw(faiss).html
    * hnsw(nmslib): https://ann-benchmarks.com/hnsw(nmslib).html
    * hnswlib: https://ann-benchmarks.com/hnswlib.html

## ANN Benchrmask における HNSW の比較

### faiss vs hnswlib

Recall/QPSは hnswlib のほうが多い≒速い。2倍まではいかないが1.5倍くらい。

同recallにおけるbuild timeは hnswlib のほうが速い。約2倍くらい。

同recallにおける index size は若干faissのほうが優秀

Relative Error/Query per second 読み方が不明。
同程度のRelative Errorのときに、hnswlibのほうが若干速そう。

### pgvector vs faiss

Recall/QPSはfaissのほうが40～50倍良い。
データ次第では差が開く。

ビルドタイムはpgvectorが2～10倍速い。

pgvector は IVF かも。なのでフェアな比較にならなそう。

2023/09 に [pgvectorをHNSWで計測するPR](https://github.com/erikbern/ann-benchmarks/pull/463) がマージされてる。Webで見れる結果に未反映、かもしれない。

FaissがIVFとHNSWを比較したベンチマークから考えれば、
HNSWにすることで検索は10倍ほど速くなる可能性がある。
すると実質的な速度は数倍になりそう。

### faiss vs nsmlib

Recall/QPSはnsmlibが速い。ただしRecallは悪そう。

Recall/Build timeは似たようなもの。

Recall/Index sizeはnsmlibのほうが大きい

nsmlibは速いが、recallに大きく欠ける印象。

### 中間まとめ

Web上で公開されているグラフは、2023/04の計測である可能性が高い。
必要な比較が公開データではできないので、自前で実行した方が良いかもしれない。

faissのHNSW実装にはメモリ効率と速度とrecallのバランスを取ろうという意図があるかもしれない。

## 最新の ANN Benchmarks による計測と考察

計測の手順は [ANN Benchmarks on local](../annbenchmarks-on-local/) を参照。
計測の結果は <https://koron.github.io/techdocs/annbenchmarks-on-local/results-20240314/> を参照。

アルゴリズムは hnsw(faiss), hnsw(vespa), hnswlib, pgvector, voyager の5つ。
データセットは glove-100-angular と fashion-mnist-784-euclidean の2つ。


### 考察: glove-100-angular

<https://koron.github.io/techdocs/annbenchmarks-on-local/results-20240314/glove-100-angular_10_angular.html>

* 全体的に Faiss の HNSW が好成績
* pgvector のQPSは同recallにおいて約10倍遅い
    * 遅いためそもそもデータ計測数が少ない(Mのバリエーションが小さい→多層グラフになる)
    * 直接のパフォーマンス比較にはならない
* pgvectorはビルド速度に優れる
* 同recallにおけるインデックスサイズは vespa が最も小さい
    * しかし高recall帯ではその差が縮まる
    * pgvectorが健闘している
* hnswlibはrecallが上がり切らない(最大約0.959)
    * Faissは最大約0.997

### 考察: fashion-mnist-784-euclidean

* MNIST: 手書き数字の画像認識データセット
* Fashion-MNIST: ファッションアイテムの画像に置き換えたもの
* recallの下限が高い。元々のデータのクラスタリング傾向が強いことに拠ると推測できる
* その他の傾向は glove-100-angular と同等

## 総評

* まずFaiss (HNSW) で良さそう (メモリに乗り切る前提)
* メモリに乗りきらない場合
    * PQなどでベクトル及びインデックスを圧縮する(recallは犠牲になる)
    * pgvectorはビルドもクエリも約10倍遅いのでそれが許容できるなら
* [ANN Benchmarks on local](../annbenchmarks-on-local/) の「考察」により詳しい考察を書いた
