# Two Tower model利用に際しての、アレコレ周辺調査

* Embeddingをクラウドで計算させたとして、取り出せるの?
    * いくらくらいかかるの?
* 手元でやるとして規模感は?
* rating(queryとitemの連結要素)の使い方についてなんかある?
    * タワー間で情報の共有
    * 現状の実装は?

## Embeddingをクラウドで計算させたとして、取り出せるの?

極論TensorFlowがあればTwo Towerモデルは作れる。
その意味でGoogle TPUが使えればそれでよい。
価格もそこがボトムラインになると推測される。

純粋にCPUで計算させた時にどうなるかは気になる。
(だいたいGPUと変わらないはずだが)

> 一般的なCPUやGPUを用いてNNの計算処理を行う場合と比較して、15〜30倍の性能と、30〜80倍の電力性能比

お? 大きく出たな。

[TPUの電力効率はGPUの何十倍も良いのか？](https://qiita.com/nishiha/items/32276cc77b27383c825f)

やや誇大広告感を感じる。

Vertex AI Matching Engineだと取り出す意味はないかもしれない。

[Two-Tower 組み込みアルゴリズムを使用してエンベディングをトレーニングする](https://cloud.google.com/vertex-ai/docs/matching-engine/train-embeddings-two-tower?hl=ja) によればやはり個別にエンベディングの学習ができそう。
ここの例ではCPUによる学習っぽい。

[Vertex AI Matching Engineの価格概要](https://cloud.google.com/vertex-ai/pricing?hl=ja#matchingengine)
よくわからない。が、読み込んでいけばおおよその価格感はつかめるかも。

[Scaling deep retrieval with TensorFlow Recommenders and Vertex AI Matching Engine](https://cloud.google.com/blog/products/ai-machine-learning/scaling-deep-retrieval-tensorflow-two-towers-architecture?hl=en)
TensorFlow Recommendersを使えばTwo-Tower作れる。

Google には Embedding 用のAPIがある。

* https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings
* https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-multimodal-embeddings
* https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/batch-prediction-genai-embeddings

そうでなければ自分で作れ、と。
その方法の1つとしてTFRSのTwo-Tower modelがある。

APIを使うなら2番目のmultimodal embeddingがフィットしそう。
multimodal = テキストと画像。
テキストは英語のみ32 wordsまで。
画像は20MBまで。
120call/minの制限あり。
価格は下記参照 (<https://cloud.google.com/vertex-ai/pricing> より)

```
Multimodal Embeddings: Text   Generate embeddings using text as an input   Text   Embeddings  $0.0002 / 1k characters input
Multimodal Embeddings: Image  Generate embeddings using image as an input  Image  Embeddings  $0.0001 / image input
```

(Vertex AI) Vector Search に名称が変更になってるらしい。
https://cloud.google.com/vertex-ai/docs/vector-search/overview

## 手元でやるとして規模感は?

そもそも2つのMLP(多層パーセプトロン)が必要。
m次元でn層の場合 `O(m^2 * n)` 程度のメモリが必要になる。
必要な次元数と層数については入力データに依存するので一概には言えないが、
(定式化はできない)
入力の次元が増えれば増えるほど必要になる。

MovieLensの例では64次元128層(=2^19 ≒約500Kワード)と推測される。
これはコードを見ることで確認できるだろう。 (←TODO)

1つ1つのデータについて、タワーを通して得られたベクトルを比較し、
差が小さくなるように誤差を逆伝搬させて学習していく、のが基本方針。
誤差の総和が評価対象なので、全データをオンメモリにする必要はないはず。

なので必要なメモリで考えるとMLPのサイズになるだろう。
証拠が必要な場合、どうやって観測しようか?

今まで試してた例は2層(128+64)かもしれない。
mm(merlin model)のMLPBlockで `[128, 64]` という指定は
各層の次元数であることはほぼ確定。

このあたりはTensorFlowのコードの基本がわかってないので
抑えるのに時間がかかりそう。
TensorFlowでCPUを使ってTwo-Towerモデリングするコードを確認する方が良いだろうか?

## rating(queryとitemの連結要素)の使い方についてなんかある?

<two-tower-model.md> より引用

> 課題: タワー間の接続が無いからパフォーマンスがあがらない
> 
> 以下のような発展形が提案されてる
> 
> * Dual Augmented Two-Tower Model (DAT)
> * Interaction Enhanced Two Tower Model (IntTower)
> * Alternative: COLD, FSCD

このあたりを手掛かりにみるのがよさそう。

[DATの概要](https://blog.reachsumit.com/posts/2023/03/two-tower-model/#dual-augmented-two-tower-model-dat)

DATの提案は2021の論文
IntTowerは2022の論文

DATは後の研究で効果が限定的だと指摘されている。

その他の研究と資料

* [Interaction Enhanced Two Tower Model (IntTower)](https://arxiv.org/pdf/2210.09890.pdf)
* IntTowerの実装 https://github.com/archersama/IntTower
* [SENET model proposed in the “Squeeze-and-Excitation Networks”](https://arxiv.org/pdf/1709.01507.pdf)
* [論文メモ: Squeeze-and-Excitation Networks](https://qiita.com/daisukelab/items/0ec936744d1b0fd8d523)
* [【論文紹介】地味にすごいSqueeze and Excitation Networks](https://qiita.com/cinchan/items/831953e32ce16c39d71b) コードへのリンク多し

Two-Towerモデルの亜種ならIntTowerが良さそう。

# (中間)まとめ

* Embeddingをクラウドで計算させたとして、取り出せるの?

    できる。APIを使うか、自前でTensorFlow等を使って作るか。

* 手元でやるとして規模感は?

    当初思ってたよりもたよりも小さく済みそう。
    ただCPUでやるのは、データが多いと現実的ではない。
    めちゃくちゃ大きなVRAMが必要ではない。
    ただVRAMに全部乗っちゃえば、学習は速くはなりそう。

* rating(queryとitemの連結要素)の使い方についてなんかある?

    (not yet)
