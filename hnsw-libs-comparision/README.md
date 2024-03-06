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

## ann-benchrmask における HNSW の比較

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

### 総評

Web上で公開されているグラフは、2023/04の計測である可能性が高い。
必要な比較が公開データではできないので、自前で実行した方が良いかもしれない。

faissのHNSW実装にはメモリ効率と速度とrecallのバランスを取ろうという意図があるかもしれない。
