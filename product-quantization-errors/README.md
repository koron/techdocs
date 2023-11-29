# Product Quantization (PQ)を用いた際の精度・誤差についての検討

## 事前知識

起点となる論文:

* [直積量子化を用いた近似最近傍探索 松井勇佑 国立情報学研究所 2016(PDF)](https://yusukematsui.me/project/survey_pq/doc/prmu2016.pdf)
* [ショートコードによる大規模近似最近傍探索 松井勇佑 国立情報学研究所 2016(PDF)](https://yusukematsui.me/project/survey_pq/doc/ann_lecture_20161202.pdf)

PQとは: 高次ベクトルを低次の部分ベクトルに分割し、各部分ベクトルをk-meansで近似することで圧縮する手法。
部分ベクトルの次数はすべて揃えるのが一般的。つまり部分ベクトルの次元数は元の高次ベクトルの次元数の約数の1つとなる。

k-meansはデータ(ベクトル)をk個に分類するクラスタリング手法。
まず全データをランダムにk個に分類する。
次に各クラスタの重心を求め、各データについて最も近い重心のクラスタに付け替える。
この重心の計算とデータのクラスタへの付け替えを、重心が変わらなくなるまで繰り返す。
初期状態でクラスタリング結果が変わるが、高速かつそこそこのクラスタリング結果を出すということで、重宝されている。

k-meansにおいては、データが与えられた時の適切なkの決め方が論じられることが多い。
一方でPQにおいては、kの値は2の累乗に固定される。
高次ベクトルを表現する際に、単なるビット列の連結で表現できるから。

## 問題意識

以上の情報からPQの精度は、分割する次元数および各分割におけるクラスタ数(k)に依存する。

また最もナイーブには、k-meansのクラスタリング精度にPQの精度が強く依存するとも言える。
k-meansの分割においては、次元数d、クラスタ数k、サンプル数nが精度に関わってくると考えられるが、その3点をまとめて論じた文献は見つけられていない。
一般的には、具体的データが与えられた時の適切なkの決定の仕方が論じられている(エルボー法、シルエット分析)。

[参考: k-meansの最適なクラスター数を調べる方法](https://qiita.com/deaikei/items/11a10fde5bb47a2cf2c2)

またk-meansに限らず、この手のクラスタリングには注意点がある。
代表的なものとしては「次元の呪い」と言われるもので、次元数dが高いほどデータが球面上に集中する現象が起こる。
これは球の半径が単位距離だけ増えた場合に、元の球の体積と、距離が増えたことにより増加した体積では、次元が高くなるにつれ後者が大きくなることによる。

[参考: クラスタリング (クラスター分析)](https://www.kamishima.net/jp/clustering/)

そもそも起点論文では、Optimized PQのようにPQする際の精度を上げるため事前変換(回転)する手法や、
各次元を複数コードブックで表すTree quantizationのような手法が紹介されている。
そちらにPQの精度や誤差について評価がされていると期待できる。

* 事前変換

    実質同じものらしい
    * [Cartesian k-means \[Norouzi CVPR 13\]](https://www.cs.toronto.edu/~fleet/research/Papers/ckmeans-CVPR13.pdf)
    * [Optimized PQ \[Ge, CVPR 13\] \[Ge, TPAMI 14\]](https://kaiminghe.github.io/cvpr13/index.html)
* 複数コードブック
    * [Optimized Cartesian k-menas \[Wang, TKDE 14\]](https://arxiv.org/pdf/1405.4054.pdf)
    * [Tree quantization \[Babenko, CVPR 15\]](https://ieeexplore.ieee.org/abstract/document/7299052)

Cartesian k-meansにはOPQのわかりやすい模式図があった。
OPQにはPQの精度評価についてそれっぽい解説と解決方法が述べられていた。

*   量子化歪み (Quantization Distortion)
*   ガウシアン分布と仮定したときのQDの下限の解析的定式化

Distortionは Optimized Cartesian k-means にも登場している。
厳密に同じものかは不明。
QDは日本語だと [量子化誤差](https://ja.wikipedia.org/wiki/%E9%87%8F%E5%AD%90%E5%8C%96%E8%AA%A4%E5%B7%AE) という方が一般的かも。

## PQそのものの一般化

以下の2つは立式が同じ。
高次ベクトルを分けずに全次元のベクトルでコードブックを作成し、その0/1の和で各ベクトルを近似する。

* [Additive quantizer](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Babenko_Additive_Quantization_for_2014_CVPR_paper.html)
* [Composite quantization](https://arxiv.org/abs/1712.00955)

[AQはFaissに実装されていそう](https://github.com/facebookresearch/faiss/wiki/Additive-quantizers)。

## まとめ

* 量子化歪み(量子化誤差)という枠組みでPQの精度・誤差を論じることはできそう
* その誤差からより良い事前変換(回転)を求めるのがOPQ
* PQを一般化した Additive quantization と Composite quantization があり、精度の議論もより一般化されるかも

突き詰めるとPQ特有の誤差というものは無くなり、普通の量子化誤差で論じることができそう
