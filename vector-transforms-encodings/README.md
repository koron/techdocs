# Vector Transforms and Encodings

## Memo

Faissより

https://github.com/facebookresearch/faiss/wiki/The-index-factory#vector-transforms

Transforms:

*   PCA (Principa Component Analysis: 主成分分析) - 出力:64次元固定
*   OPQ16 - 出力:d次元もしくは64次元固定
*   RR (Random Rotation: 乱雑に回転…) - 出力:64次元固定
*   L2norm (L2正則化: 過学習の抑制) - 出力:d次元
*   ITQ (Iterative Quantization)- 出力:256次元もしくはd次元
*   Pad (次元を増やして0で埋め128次元にする) - 出力:128次元固定

よくわかってないのは OPQ, L2norm, ITQ の3つ。
わかってるのは PCA, RR, Pad

Encodings:

*   Flat
*   PQ16 (Product Quantization)
*   PQ28x4fs (PQ + fast scan (SIMD))
*   SQ (Scalar Quantizer)
*   Residual (Residual Encoding)
*   RQ (Residual Quantizer)
*   LSQ (Local Search Quantizer?)
*   PRQ (ベクトルを16個に分けて、それぞれをRQ)
*   PLSQ (16個に分けてLSQと同様?)
*   ZnLattice (球面上の格子?)
*   LSH (Locality Sensitive Hashing: 各ベクトルを2値化?)
*   ITQ90,SH2.5 (ITQ + Sensitive Hashing?)

SQ, RQ, ZnLattice, LSH あたりが不明ポイント。
時点でRQ, LSQ などのバリエーションは、ほんとにバリエーションなのか?
Flat, PQは理解できてる(しやすい)。

Transformsには2つの性質がある(1つしかない場合もある)。
1つはEncodingの精度を上げる性質。
もう1つは次元を圧縮する性質。

Encodingsは次元を圧縮する。
元データの分布を利用したものと、そうでないものという分け方はできそう。
PQは汎用で、ZnLatticeやLSHは特殊用途。

OPQ: [Optimized Product Quantization for ANN search (2013)](https://ieeexplore.ieee.org/document/6619223)

<https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/pami13opq.pdf>

PQによる歪みの最小化。
16個のサブベクトルに分けて事前に回転し、PQの精度を高める。
出力時点で64次元に圧縮もできるが、元の次元に戻すのもあり。
どうやって回転(optimize)してるのかは不明だが、
まぁやることはわからなくはない。

ITQ: <https://slazebni.cs.illinois.edu/publications/ITQ.pdf>
詳細を理解するには論文に目を通す必要がありそう。
ベクトルをLSHでエンコードする際に便利。
CCA (Canonical Correlation Analysis) = 正準相関分析
<https://qiita.com/yoneda88/items/847cb99542538083b876>
主に画像用途。

L2norm: L2正則化は過学習の防止が目的。
損失関数にL2ノルム(二乗和の平方根≒距離)を加えることで正則化する。
<https://zero2one.jp/ai-word/l2-normalization/>
<https://zero2one.jp/learningblog/yobinori-collab-regularization/>
ヨビノリの解説動画がある。あとでゆっくり見ておこう。
