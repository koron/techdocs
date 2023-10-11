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

* MercariがGoogleのVertex AI Matchingを使って
* 個人がTensorflow Recommendersを使って
* 英語圏では?
    * 当然 YouTube 他のGoogleサービスでは使ってる
    * Uber: <https://www.uber.com/en-JP/blog/innovative-recommendation-applications-using-two-tower-embeddings/> プラットホームは不明
    * <https://www.linkedin.com/pulse/personalized-recommendations-iv-two-tower-models-gaurav-chakravorty/> Facebook の Video Recommendation に関わってるらしい人の解説 (2021/01/24)
    * Snapchat Spotlight: <https://eng.snap.com/embedding-based-retrieval> 2023/06/06 一部にGoogleを使ってるらしいが、コア部分であるとは断定できない
        * **EBR (Embedding-based Retrieval)** というキーワード
    * Tencentの論文 https://arxiv.org/abs/2302.08714 2023/02/17
    * Facebookの論文 https://research.facebook.com/publications/embedding-based-retrieval-in-facebook-search/ 2020/08/22
    * Airbnbの論文 https://arxiv.org/pdf/2004.02621.pdf
    * Instagramでどう使ってるのか https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/ 2023/08/09
    * Alibabaの論文 [ATNN: Adversarial Two-Tower Neural Network for New Item’s Popularity Prediction in E-commerce](https://personal.ntu.edu.sg/c.long/paper/21-ICDE-arrivalPrediction.pdf) タイトルからは Triplet Lossっぽい雰囲気がある (時期不明 2021 ICDE? 04?)
* 気になった資料
    * EBRとは? (Embedding-based Retrieval) https://ymym3412.hatenablog.com/entry/2020/12/10/035027
    * EBRの論文読み祭り https://speakerdeck.com/stakaya/tui-jian-ge-she-he-tong-lun-wen-du-miji-number-1-kdd-20-embedding-based-retrieval-in-facebook-search
    * Google [re-training Tasks for Embedding-based Large-scale Retrieval](https://research.google/pubs/pub49252/) 2020
    * Google [Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations](https://research.google/pubs/pub50257/) 2020
    * <https://github.com/liyinxiao/UnifiedEmbeddingModel> Facebookの論文を実装してみた系 two-tower x2の入れ子構造になってる。 torch & numpy
    * [ZOZOでの検索/推薦技術に関する論文読み会](https://techblog.zozo.com/entry/search-recommend-articles-study-session)
    * [Deep Metric Learning の定番⁈ Triplet Lossを徹底解説](https://qiita.com/tancoro/items/35d0925de74f21bfff14)
        * Triplet Loss 2014/04 ポジティブとネガティブ、組で学習する点がポイントみたい
    * [Beyond Two-Tower Matching: Learning Sparse Retrievable Cross-Interactions for Recommendation](https://dl.acm.org/doi/abs/10.1145/3539618.3591643)

## その他

### NVTabularが「GPUモードではない」というのはなぜ? 回避法は?

(dockerを使うことで解決済み)

cudfパッケージがロードできないことによる。

cudfはconda経由でインストールしろと怒られる。

pypiには古い版しかなさそう。

Windowsではインストールが困難かもしれない。

RAPIDSの一部。

> RAPIDS はデータサイエンスのワークフロー全体を GPU で高速化するためのライブラ
> リ群です。GPU の性能を引き出す NVIDIA CUDA ベースで構築され、使いやすい
> Python インタフェースを提供します。

<https://qiita.com/Clip_1212/items/beaad136216c18a1d2a4>

Windowsは未サポート情報アリ。

これ以上頑張る意義がないかも。

WSL2でやる選択肢と、dockerでやる選択肢がある。
他パスでもDockerに行き当たってるので、いまはDockerが優先だろう。
とにかく誰でも動かせるベースラインを作ることが目的。
(ただしダウンロード量はアホみたいに多いがw)

### on Docker

```
docker run -it --rm --gpus all \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8888:8888 \
    -v .:/workspace/data/ --ipc=host \
    nvcr.io/nvidia/merlin/merlin-tensorflow:nightly \
    /bin/bash
```

これでDockerを起動して、シェル内で以下のコマンドでjupyter-labを動かせば、
以前動かなかったサンプルも動作を確認できた。

```
jupyter lab --allow-root --ip='0.0.0.0'
```

イメージの詳細(カタログ)
<https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow>

全イメージで14.6GBになるので、ディスク容量には注意。

未ログインでもpullはできたらしい。

## Two-Tower modelについての理解

ユーザーがアイテムを購入したというデータがある。

ユーザーとアイテムをそれぞれのDNNタワーにかけ適当なembeddings(ベクトル)を得る。
ユーザーAがアイテムBを購入した場合、その2つのベクトルが近くなるように学習する。
ここに損失関数を導入する。
Googleのは鼻薬が効いてて、多く登場するアイテムが過度に有利・不利にならないようになってる。

全購入データの2つのベクトルの距離が小さくなる方向に圧力をかけることが学習になる。
裏返すと、買わなかった組み合わせでは距離が大きくなる方向に圧力をかける。

追加資料: <https://blog.reachsumit.com/posts/2023/03/two-tower-model/>

課題: タワー間の接続が無いからパフォーマンスがあがらない

以下のような発展形が提案されてる

* Dual Augmented Two-Tower Model (DAT)
* Interaction Enhanced Two Tower Model (IntTower)
* Alternative: COLD, FSCD

## ゼロベースからの再現

movieslensのデータを使ってTwoTowerを構築し、簡単な検索を実施するところまで。

`merlin.datasets.entertainment.get_movielens` を使うと楽。
ただ一般的なデータは読み込めないので、
どうやって読み込むかについて一般かが必要。
最悪 `get_movielens` の実装を読めばよい。

movielens 100kと1mではスキーマのカラム数が異なる。
1mのほうが多く、主に `TE_` プレフィクスのカラムが増えてる。
っていうかダウンロードできるzipの構成が全然違う。

`get_movielens` は結構複雑なことをやってる可能性があり。
zipのダウンロード、展開、必要なデータの読み込み、
必要なタグの付与、
parquet化して保存、Data Frameでの読み込み。

入力ファイルは `movies.dat`, `ratings.dat`, `users.dat` の3つで
出力ファイルは `movies_converted.parquet`, `train.parquet`, `valid.parquet`, `users_converted.parquet` の4つ。
映画作品のデータ、ユーザーのデータ、ユーザーが映画を視聴して付けた点数データという感じ。
点数データに映画作品とユーザーのデータを非正規化して合成し、8:2でtrainとvalidに分けるいう流れ。

当然 train と valid のスキーマは一致する。
そのスキーマのうち `Tags.USER` でマークされたものを新たなユーザスキーマとして、
入力フィルタにし two-tower のうちquery 側のエンコーダーにする。
エンコーダーはMLP(多層パーセプトロン)である。

同様のことを `Tags.ITEM` でマークされたスキーマに対して行い、
candidate 側のエンコーダーにする。

queryとcandidateのエンコーダーを指定して TwoTowerModelV2 を作成し、
コンパイル (`compile`) し、train を用いて `fit` する。
この時のパラメーターについては、別途調べる必要がある。

train から `Tags.ITEM` および `Tags.ITEM_ID` で特徴量(?)を抽出すれば、
それを元にTop-Kを求める特殊なエンコーダー(`topk_model`)が作成できる。
validの任意のitem(映画作品)を指定してそれに近い作品を提示できる、ということ。
ここで言ってる特徴量とは `features` で `unique_rows_by_features(source?, columns?, key?)` により求められるはず。

`unique_rows_by_features` についてはわかってないことが多いので
ソースを見た方が良いかもしれない。
非正規化データから正規化データを取り出していそうな雰囲気。

ネガティブサンプリングに基づく、モデルの評価といえるらしい。
ネガティブサンプリングってなんじゃらほい?

<https://benrishi-ai.com/negative-sampling01/>
<https://kento1109.hatenablog.com/entry/2019/11/29/111028>
<http://tkengo.github.io/blog/2016/05/09/understand-how-to-learn-word2vec/>

もともとはサンプルに存在しない組み合わせを、学習へ利用することで高速化する手法らしい。

評価方法がよくわからない。

modelから `query_encoder` を取り出してシリアライズできる。
となるとシリアライズしたデータか model を学習をせずに再構築するのもアリのはずだが。

1. データのロード (非正規化)
2. カラムのタグ付け `Tags.USER_ID`, `Tags.USER`, `Tags.ITEM_ID`, `Tags.ITEM`
   スキーマの分離が目的であって、必ずしもこのタグである必要はないはず。
3. queryとcandidateのエンコーダを作成(2によりスキーマを分類し、フィルタとする)
4. TwoTowerモデルの作成と学習 (1全体の読み込み)
5. 上記モデルの評価 (Negative SamplingによるTop-Kの評価)
6. embeddingsの取得


### TODO: 要調査アイテム

* `get_movielens` の中身
* `TwoTowerModelV2` の `compile` と `fit` のパラメーター
* `unique_rows_by_features` の中身
* シリアライズデータからモデルを構築する方法?


## 参考資料

* <https://scrapbox.io/pokutuna/Two-Tower>
* [Tensorflow Recommendersの精度を上げるためのテクニック](https://zenn.dev/yng/articles/improving_tfrs_accuracy): 実際に導入する際には熟読すべし
