# Vertex AI Matching Engine

## 概要

<https://cloud.google.com/vertex-ai/docs/matching-engine/overview?hl=ja>

* Model Archtecture (embedding)
    オリジナルベクトルを embedding 空間にマッピングする

    * Database Tower
    * Query Tower

* Embedding Space

    近似空間

考慮すべきポイント

* Database Towerをどう得るか
* Query Towerをどう得るか
* Embedding Spaceでの近傍探索の最適化

アピールされてるポイント

* アイテムのエンベディング表現を生成する 
* エンベディングに対して最近傍探索を実行する

モジュール化されているので、他サービスと組み合わせて使うこともできる

## Train embeddings two tower

<https://cloud.google.com/vertex-ai/docs/matching-engine/train-embeddings-two-tower?hl=ja>

どういうフォーマットを受け付けるか。
トレーニングツールのチュートリアル。オプションとか。

## Matching Engine ANN サービスの概要

<https://cloud.google.com/vertex-ai/docs/matching-engine/ann-service-overview?hl=ja>

> 10 億以上のベクトルのコーパスを数ミリ秒で検索できます

Matching Engine ANNで何が提供されるかの解説。

## Create and manage index

<https://cloud.google.com/vertex-ai/docs/matching-engine/create-manage-index>

1. インデックスを設定する(次元とか距離関数とかの設定)
2. 実際にインデックスを作る

    インデックスのデータは複数のShardに分割される。サイズに目安が与えられてる。
    バッチ更新かストリーム更新かで、適切なインデックスの作り方が異なる
3. インデックスを一覧する
4. インデックスのチューニング

    作った後に一部はいじれるっぽい。
5. インデックスを消す

## インデックスの構成

<https://cloud.google.com/vertex-ai/docs/matching-engine/configuring-indexes?hl=ja>

index configuringについての日本語解説。

tree-AH アルゴリズムってなんだ?

## Query indexes to get nearest neighbors

<https://cloud.google.com/vertex-ai/docs/matching-engine/query-index-public-endpoint>

## Scaling deep retrieval with TensorFlow Recommenders and Vertex AI Matching Engine

> tree-AH アルゴリズムってなんだ?

<https://cloud.google.com/blog/products/ai-machine-learning/scaling-deep-retrieval-tensorflow-two-towers-architecture?hl=en>

> When creating an ANN index, Matching Engine uses the Tree-AH strategy to build a distributed implementation of our candidate index. It combines two algorithms 

2つのアルゴリズムを組み合わせたもの。

* 階層的なクラスタリング・ツリー構造
* クエリベクトルとノード間の近似度としての、内積の高速な近似のための非対称なハッシュ

HNSW+PQって感じ

## ここまでの感想

ANNを、細かい設定は抜きにして、実用的な速度で使いた人向け。
モードはBrute ForceかHNSW+PQ的なものの2種類。
あとはサイズで全部決まる。
技術詳細は隠されているが、性能面の保証(?)がされており、
「お金を出せばできるよ」的でありこれはこれで納得度がある。

こっから先は、実際に動かすコードをみたほうがよいかも。

## メルカリの例

<https://engineering.mercari.com/blog/entry/20220224-similar-search-using-matching-engine/>

* Matching Engine導入の動機 ← 納得できる
    * idが文字列である(Faissは数値で、自前のマッピングが必要)
    * indexの更新がmanaged service
* word2vecを使って、自前でembedding化

具体的は使い方については特に解説がない。
(これで十分・これ以上は出せないという感じはある)

まぁFaissベースよりは全然楽そうだよね。

## AWS managed service

ならAWSにもありそうなんだけど…

<https://docs.aws.amazon.com/ja_jp/opensearch-service/latest/developerguide/knn.html>

OpenSearchとElasticsearchでやってる。
Elasticsearchは前に調べた通り、速度とか最適化具合がいまいちのハズ。
スケール感が違うから。

OpenSearchはどうなんだ?

<https://ja.wikipedia.org/wiki/OpenSearch>
<https://www.designet.co.jp/ossinfo/opensearch/>

Elasticsearchのライセンス変更に伴い、フォークして生まれたっぽい。

本質的にはElasticsearchと同じと考えて良いだろう。

## サンプルノート

<https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/matching_engine/matching_engine_for_indexing.ipynb>

セットアップが長い。
データはHDF5から事前変換してる。
Matching EngineがJSON LinesもしくはCSVかAVROだからしかたない。

tree-AHのインデックスと、Brute forceのインデックスを作ってる。
Brute forceのインデックスは、ground truth即ち検証用らしい。

初期データの導入がよくわからないが、incrementalで入れてる箇所はあった。

VPCネットワーク向けのエンドポイントを開く

インデックスをデプロイ。

クエリ実行。バッチクエリ実行。

recall指標の計算。
Brute force と tree-AHで比較して計算してる。

クリーンアップ。

## まとめ

とにかく楽に使う(使わせる)ことを考えたサービス。
アルゴリズムはBrute forceとtree-AHだけの選択で、選択肢が少ない分だけ迷わない。
複数shardを使う前提。
そこそこのお金で解決するモデル。
