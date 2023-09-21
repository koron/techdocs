# Two Tower Modelとは?

* どこからきた? 誰が考えた? 論文は?
* NVIDIA-Merlinでの位置づけは?
* 先行利用者は?
* その他

## どこからきた? 誰が考えた? 論文は?

Two Tower Modelの原型となった論文

Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations
(大きなコーパスのアイテム推薦のための、サンプリングバイアス修正神経モデリング)

<https://research.google/pubs/pub48840/>

PDF:
<https://storage.googleapis.com/pub-tools-public-publication-data/pdf/6417b9a68bd77033d65e431bdba855563066dc8c.pdf>

被引用論文:
<https://dl.acm.org/action/ajaxShowCitedBy?doi=10.1145/3298689.3346996>

行列の因数分解でやるところを、Two-Towerニューラルネットを用いたモデリングでやる。
このモデルの学習は損失関数の最適化だが、
サンプリングによる学習では偏りがありモデル性能の劣化が懸念される。
この研究ではそこにメスを入れた。

別実装:

* https://www.tensorflow.org/recommenders?hl=ja
    * 利用例: https://www.tensorflow.org/recommenders/examples/basic_retrieval?hl=ja

`MovielensModel` もしくは `MovieLensModel` で検索すると、もうちょい情報が出てくる。

* [TensorFlow Recommendersはいいぞってこと！](https://qiita.com/TsuchiyaYutaro/items/d5f90cb10b490ef9f223)
* [Tensorflow Recommendersの精度を上げるためのテクニック](https://zenn.dev/yng/articles/improving_tfrs_accuracy)

Two-Tower model自体はGoogleの論文以前からあった。
Googleの論文は既存Two-Towerモデルの弱点(人気アイテムほど負例にもなりやすい)を改善するもの。
TensorFlow RecommendersのTwo-Towerは古いもので、
Merlinのは新しいものである可能性が高い。

## NVIDIA-Merlinでの位置づけは?

現実的なExampleは <https://github.com/NVIDIA-Merlin/Merlin/blob/main/examples/Building-and-deploying-multi-stage-RecSys/01-Building-Recommender-Systems-with-Merlin.ipynb> が最有力。

NVIDIA-Merlin内でその他に使ってるのはExampleくらい。

数あるモデルの1つ。RetrievalModelおよびV2のバリエーション。
クラス階層図
```
tensorflow.keras.Model
  BaseModel
    Model
      RetrievalModel
        TwoTowerModel
      RetrievalModelV2
        TwoTowerModelV2
```


## 先行利用者は?

## その他

### NVTabularが「GPUモードではない」というのはなぜ? 回避法は?

## 参考資料

* <https://scrapbox.io/pokutuna/Two-Tower>
* [Tensorflow Recommendersの精度を上げるためのテクニック](https://zenn.dev/yng/articles/improving_tfrs_accuracy): 実際に導入する際には熟読すべし
