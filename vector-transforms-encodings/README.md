# Vector Transforms and Encodings

## Memo

Faissより

<https://github.com/facebookresearch/faiss/wiki/The-index-factory#vector-transforms>

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

損失関数で与えられるエネルギーの等高線に対して、
円(≒L2ノルム)を与えてその円上の最小地点を得ることに相当するらしい。
ちなみにL1ノルムは正方形になり、代替の場合その頂点となるとのこと。

SQは浮動小数点を整数にエンコードすること。
均等な幅にすることもできるが、不均等な幅でエンコードすることもできる。

Residual Encoding は「残差」とかそんな感じだろうか?
複数の重心に分解して表す。より正確にするには残差をPQやSQで表す必要あり。
<https://www.sciencedirect.com/topics/engineering/residual-coding>
古く(2016)はオーディオ系に見られる。
n個の重心を選び、いずれかからの残差をPQやSQで圧縮することで精度とサイズのバランスを取る。
MI centroidという言葉もでてきたが、よくわからない。
MI≒慣性モーメントとする説明が出てきたが、慣性モーメント重心となるともっとわからない。

Twitterが用意した、FaissをJavaから呼び出すJNIモジュールを見つけた。
<https://github.com/twitter/the-algorithm/tree/main/ann/src/main/java/com/twitter/ann>

RQはResidual quantizer codec
2^16の重心で量子化(分類)したのち、残差を256の重心で6ステージエンコードする。
言い換えると複数のコードブック(代表ベクトル)の組み合わせで近似する。

<https://www.assemblyai.com/blog/what-is-residual-vector-quantization/>

ZnLattice. 格子コーデック。
Zn格子 <https://ieeexplore.ieee.org/document/10131006>

以下、部分的な情報からの推測。
単位球面上に格子を形成して各点にID整数値を紐づける。
ベクトルをその整数値+長さで表現するのがZnLattice。
実際には1つのベクトルを分割し、それぞれをエンコードするということが考えられる。

LSH.
各要素を閾値で二値化する。
suffixの `r` は事前に回転させる。
suffixの `t` は閾値を学習で決定する。
ITQと併せて使うらしい。

ITQ再訪。Iterative Quantization.

SONYの解説: 量子化によるニューラルネットワークのコンパクト化
<https://www.youtube.com/watch?v=qpd9I8m1bOA>

PCAやCCAがより有効になるような、事前の回転を求めるのがITQということらしい。
回転を探索する指針としては、超立方体の各頂点にクラスタの代表点を配置する。

なおPQはベクトルを同じサイズの部分ベクトルに分け、それぞれを個別に量子化。
(単なる復習として書いた)

## 所感

圧縮のためにはベクトル量子化が避けて通れない。

極端な量子化手法は超立方や超球体上での表現になる。
それに適した事前変換(主に回転)が効果的に機能する。
これらは求める分類にそういう高次の構造がある場合にはよく機能する。

求める分類が未知(予測不可)の場合、数値的に適応的に量子化するしかない。
その観点では、決まったサイズで分けて適応的に量子化するPQ、
および分けた単位で事前に回転を行うOPQは、
圧縮率は他にくらべてイマイチだが納得はできる。
