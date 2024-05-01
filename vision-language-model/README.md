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
