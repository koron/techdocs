# IVF vector search とはなんぞや?

IVF = Inverted File Index : 転置ファイルインデックス

[転置インデックス](https://ja.wikipedia.org/wiki/%E8%BB%A2%E7%BD%AE%E3%82%A4%E3%83%B3%E3%83%87%E3%83%83%E3%82%AF%E3%82%B9) というのは聞いたことがある。
複数の文章に対しそれらの文章を構成する単語がどの文章に登場したかを記録しておくこと。
検索文の構成単語からより多くの単語が登場する文章を決定し検索結果とする。

転置インデックスと転置ファイルインデックスの関係が不明。
さらにベクトルインデックシングへどう適用できるのかが謎。

良く見たら IVF って InVerted File index という感じで変則的なアクロニムになってるのね…
どうりでなかなか頭に入ってこないわけだ。

ボロノイ図、ディリクレ空間分割 (Dirichlet tessellation) を元に検索範囲を分割することに伴う。
分割された検索範囲(セル)の中でクエリベクトルを検索するので、全体から検索するよりも速くなる。

クエリベクトルがセルの境界に近い場合、境界を越えられないため近傍点を逃す可能性が出てくる。
それを解決するために近傍のセルまで探索領域を増やす。

Faissでは `nprobe` 及び `nlist` の2つのパラメーターが指定できる。
`nprobe` = 何個の近傍セルまで探索を広げるか。
`nlist` = 何個のセルに分割するか。
つまり計算オーダーとしては全体検索に比べて $ \frac{nprobe}{nlist} $ になる。

疑問点: そもそもボロノイ分割するときの代表点の決定方法(クラスタリング)はどうしてるのだろう?

Faissでは [k平均法](https://ja.wikipedia.org/wiki/K%E5%B9%B3%E5%9D%87%E6%B3%95) だった。
参考: <https://github.com/facebookresearch/faiss/blob/dafdff110489db7587b169a0afee8470f220d295/faiss/Clustering.h#L61-L73>
イテレーションは [10回](https://github.com/facebookresearch/faiss/blob/dafdff110489db7587b169a0afee8470f220d295/faiss/IndexIVF.cpp#L46) らしい。
その他にも細かなパラメーター ([ClusteringParameters](https://github.com/facebookresearch/faiss/blob/dafdff110489db7587b169a0afee8470f220d295/faiss/Clustering.h#L21)) があるが、IVFで設定できるかは不明。
IVFFlatがLevel1Quantizerをstructで継承しているのでC++レベルで言えばクラス外から変更できるはず。

pgvector/pgvector もk平均法だった([参考](https://github.com/pgvector/pgvector/blob/c6ddf62a29f4790d386a60f9826583d2a228ef68/src/ivfkmeans.c#L547))。
Elkan's accelerated Kmeansというアルゴリズムがあるらしい。
論文 [Using the Triangle Inequality to Accelerate k-Menas](https://cdn.aaai.org/ICML/2003/ICML03-022.pdf)
距離の再計算を減らすというコンセプト。
`k'` 個のcentroidが変更されたときに再計算が必要な個数は `k' N` になる。
三角不等式を利用し、計算すべき距離を絞り込む。

## 参考資料

* <https://www.pinecone.io/learn/series/faiss/vector-indexes/#Inverted-File-Index>
