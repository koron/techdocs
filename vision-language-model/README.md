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
