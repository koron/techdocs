# 画像のembedding

## Survey

既存LLMに画像を説明させ、その文章をベクトル化し分類する。
<https://note.com/tdual/n/n7b645c66ea19>

マルチモーダル検索とは何か: 「視覚を持った LLM」でビジネスが変わる
<https://cloud.google.com/blog/ja/products/ai-machine-learning/multimodal-generative-ai-search>

CoCa: Image-Text Pre-training with Contrastive Captioners
<https://research.google/blog/image-text-pre-training-with-contrastive-captioners/>

(オート)エンコーダー: 高次元データの次元圧縮ができるニューラルネットワーク。
入力・中間・出力の3層からなるNNで、入力と出力が一致するように学習する。
このとき中間層の次元を絞る・減らすと入力&中間層のペアが、次元を圧縮するエンコーダーになる。

画像のエンコーダーとテキストのデコーダーをくっつけて学習したら、画像を説明するネットワークが得られる。
画像とキャプションのペアのサンプルがそれなりに要りそう。
だがいきなり画像→テキストの変換を学習するよりは、確かに良さそう。

画像は大きいから分割して入れる必要があるのでは?
分割した画像間に、なんらかの関係性を保持して利用できるはずでは?
それは文章とかにも関係するのでは?

### InsturctBLIPS

Paper: <https://arxiv.org/abs/2305.06500>

* [BLIPというモデルについて調べてみた](https://eng-blog.iij.ad.jp/archives/23804)
* [Japanese InstructBLIP Alpha](https://ja.stability.ai/blog/japanese-instructblip-alpha)


### LLaVA

LLaVA = Large Language and Vision Assistant (2023-Apr)

* [LLaVA](https://github.com/haotian-liu/LLaVA)
    * Paper: <https://arxiv.org/abs/2304.08485>
* [マイクロソフト、130億パラメータの言語・視覚チャットボット「LLaVA」をオープンソース化](https://www.infoq.com/jp/news/2023/07/microsoft-llava-chatbot/)
* [日本語LLMでLLaVAの学習を行ってみた](https://qiita.com/toshi_456/items/248005a842725f9406e3)

    日本語LLMをLLaVAで使えるようにする方法を通じてLLaVAの構成が明らかになる。

構造は3つのモジュールからなる

* Vision Encoder: おそらくオートエンコーダーの前半部分
* LLM
* Vision Projector: Vision Encoderの出力をLLMへ繋げる部分(コンバーター的な?)

学習方法:

1. Vision Encoder と LLMを凍結し Projector を学習
2. Vision Encoder のみを凍結し、Projector & LLM を学習

事前学習データが600Kと少ない。ファインチューニングは158K

### Others

* GPT4-V
    * [画像を理解する大規模言語モデル「GPT-4V」とは？](https://note.com/generative_ai/n/n869a74f9eb53)
* Gemini
* YOLO (You Only Look Once)
    * [【物体検出手法の歴史 : YOLOの紹介】](https://qiita.com/cv_carnavi/items/68dcda71e90321574a2b)
    * [YOLOとは？なぜ早い？物体検出の従来手法との違い・メリット・デメリットを詳しく解説](https://ai-market.jp/technology/yolo/)
* [アップルがLLMのようにスケーラブルな大規模自己回帰画像モデルを開発](https://ai-scholar.tech/articles/computer-vision/AIM)
* [ImageNet データセットとは](https://cvml-expertguide.net/terms/dataset/image-dataset/imagenet/)
    * [WordNet](https://ja.wikipedia.org/wiki/WordNet) ベース

## ImageNetから始めるデータセット

### ImageNetの課題

* モデル側の性能がサチった ≒ データ量が少ない
* バイアスが強くなった
    * ラベル誤りに対して過学習
* 複数物体の写り込みに正しくアノテーションされてない
* 人物に対する配慮不足
    * プライバシー
    * 侮辱的表現

### ImageNetの後継

* [PASS](https://www.robots.ox.ac.uk/~vgg/data/pass/)
    * [PASS: An ImageNet replacement for self-supervised pretraining without humans](https://arxiv.org/abs/2109.13228)

    * [The End Of ImageNet](https://analyticsindiamag.com/the-end-of-imagenet/)
* [FGVC-Aircraft](https://paperswithcode.com/dataset/fgvc-aircraft-1)
    航空機のデータ
* [iNaturalist 2018, 2019, and 2021](https://paperswithcode.com/dataset/inaturalist)
    自然物のデータ
* [WebVision-1000](https://paperswithcode.com/dataset/webvision-database)
    FlickrやGoogle画像検索のデータ
* [YFCC100M](https://paperswithcode.com/dataset/yfcc100m)

<https://paperswithcode.com/datasets> にいろんなデータセットの情報がまとまってそう。

そもそもラベルありのデータを使うのは時代遅れかも。

### 学習方法とデータセット

* SSL (Self-Supervised Learning)
* HPT (Hierarchical Pre-Training)
* 教師無し学習 (unsupervised method)

### 画像に対する教師無し学習

* [Unsupervised Pre-Training of Image Features on Non-Curated Data](https://arxiv.org/abs/1905.01278) (2019, Facebook)
    YFCC100Mを用いて、教師無しで事前学習する方法
* [Image Representations Learned With Unsupervised Pre-Training Contain Human-like Biases](https://arxiv.org/abs/2010.15052)
    そうやって学習したら人間に近い特徴が得られたよ
* [End-to-End Unsupervised Vision-and-Language Pre-training with Referring Expression Matching](https://aclanthology.org/2022.emnlp-main.742/) (2022)
    教師無しは外部の物体検出に依存し、制限を受ける。
* [DETReg: Unsupervised Pretraining with Region Priors for Object Detection](https://qiita.com/sasakits/items/d7c89dc7f055b2fa6152) (2021)
* [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882)
* [画像に対する自己教師あり表現学習手法について②](https://blog.recruit.co.jp/data/articles/ssl_vision_02/)
    サーベイとして素晴らしい

## Vision Transformer

* [画像認識の大革命。AI界で話題爆発中の「Vision Transformer」を解説！](https://qiita.com/omiita/items/0049ade809c4817670d7)
* [Vision Transformerのモデル構造](https://qiita.com/wakayama_90b/items/55bba80338615c7cce73)
* [画像識別(クラスタ識別)タスク](https://paperswithcode.com/task/image-classification)
