# 実践的ファインチューニング

本文章の目的はファインチューニングを実際に行ってみて、実践的な手法を確立すること。
もしくはその記録。

## TL;DR

* プロンプトを工夫する
* モデル・パラメーターの明確な評価基準を定める
* 学習はガチャ(≒良い結果を得るにはなんども学習しなおすしかない)
* 学習にかかる時間を短くし、評価を自動化し、イテレーション数を稼ぐ

## 問題設定

Web広告の画像が日本の法や条例に違反してないかどうかを判定する、
という機能を事前学習済みのVLM (Vision and Language Model)に獲得させる。

ベースとなるモデルおよびパラメーターは paligemma-3B-pt-224 ([PaliGemma](https://developers.googleblog.com/ja/gemma-explained-paligemma-architecture/)) で、
入手元は [Kaggle](https://www.kaggle.com/models/google/paligemma/jax/paligemma-3b-pt-224) である。

利用するデータは全96個で、違反してない(`OK`)ものが54個、違反している(`NG`)ものが42個となっている。
形式はPNGが45個、JPGが44個、GIFが7個でGIFはアニメーションを含んでいる。
データそのものは現時点で非公開で、公開できるかは検討中。

実行にはNVIDIAが用意しているjax:gemmaのdockerイメージ ([ghcr.io/nvidia/jax:gemma](https://github.com/nvidia/JAX-Toolbox/pkgs/container/jax)) を用いる。
環境は Windows 11 Pro + Core i9 9900K + NVIDIA RTX 4070 (VRAM 12GB) + Docker Desktop 。
VRAMが小さいためバッチ数を小さくせざるを得ず、学習が発散する確率が若干高くなっている。

### 戦略

プロンプトは `yes` か `no` を答えさせるものにし、たかだか2つのクラスタリング問題([二項分類](https://ja.wikipedia.org/wiki/%E4%BA%8C%E9%A0%85%E5%88%86%E9%A1%9E))に帰着させる。

二項分類にしたことで学習結果の評価にはAccuracy, Recall, Precision, F-measureを使える。
参考: [統計学および機械学習の評価指標#二値分類](https://ja.wikipedia.org/wiki/%E7%B5%B1%E8%A8%88%E5%AD%A6%E3%81%8A%E3%82%88%E3%81%B3%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%81%AE%E8%A9%95%E4%BE%A1%E6%8C%87%E6%A8%99#%E4%BA%8C%E5%80%A4%E5%88%86%E9%A1%9E)

データは学習用と検証用に分けて用いる。
分ける比率自体も9:1、8:2, 7:3といったように変えて傾向を検証したが、絶対数が96と多くない現在は9:1 (86:10)で良さそうだ。

極論、追加学習はガチャ(≒乱数に強く依存する)なので、
先にパラメーターの評価手法を確立した上で、
学習のたびに評価してより良いパラメーターを採用する

### 実際の手順

プログラムは [paligemma のデモ](https://ai.google.dev/gemma/docs/paligemma/fine-tuning-paligemma) をベースに適宜改造したものを用いた。

このような前提で、まずは以下の手順でプロンプトを決定する

1. 追加学習前のパラメーターに対してプロンプトを用いて、データ全体を推論させ評価値を得る
2. プロンプトを改良する: プロンプトを変更するたびに評価値をステップ1で再計算し、より良い評価値になるプロンプトを採用する

次に以下の手順で学習前後でパラメーターを評価する

1. データ全体を学習用と検証用に、あらかじめ決めた比率でランダムに分ける

    コマンド例: [JSON Lines](https://jsonlines.org/)データをシャッフルして、先頭10個を検証用に、残り86個を学習用に振り分けてる。

    ```console
    $ DIR=outdir ; $ mkdir -p $DIR
    $ shuf all.jsonl > $DIR/shuffle.jsonl
    $ head -10 $DIR/shuffle.jsonl > $DIR/valid.jsonl
    $ tail -86 $DIR/shuffle.jsonl > $DIR/train.jsonl
    ```

2. 学習前に評価値をデータ毎に得る
    * 学習後に比較検討するため
    * データ全体、学習用データ、検証用データ、それぞれ個別に評価値を得る
3. 学習用データを用いて1回目の学習する
    * 学習率は 0.03
    * バッチ数は 2
    * その他のパラメーターはデモプログラムのものをそのまま採用し固定
    * ときどき発散する(学習中にlossが100を超えるほど極端に大きくなるのでわかる)ので、その場合は学習を中止し最初からやり直し
4. 学習後の評価値をデータごとに得る
    * データ全体、学習用データ、検証用データ、それぞれ個別に評価値を得る
5. 学習済みパラメーターを保存する

学習用データに対する各種評価値が 1.0 に収束していなければ、さらに追加学習することを検討する。
追加学習の差異には学習率は 0.003 と、1回目の1/10にして発散を避けつつフィッティング(収束)を狙う。
ただしデータが増えたら、追加学習の回数を増やしたり学習率を変更したり、
学習を止めるタイミングを変更したりする必要があると推測される。

学習用データに対する各種評価値が 1.0 に収束している(もしくは学習を完了させた)場合、
学習前後の、検証用データに対する各種評価値は学習結果の品質を示していると考えられる。
基本的には増えていれば増えているほど良く、減っていれば悪い。
どの評価値をより重視するかはケースによって違うが、
法律や条例に違反してないことを重く見るのであれば、
False Positive が少ないことを示す Precision を採用するので良い。

## 工夫すべきポイント

### プロンプト

当初利用していたプロンプトは以下の通り「この画像は日本の薬事法に基づいて適切でしょうか?」という意味。

> Is this image appropriate based on Japan's Pharmaceutical and Medical Device Act?

このプロンプトで学習するということは、
英語で事前学習を済ませたモデル・パラメーターに対して
"Japan's Pharmaceutical and Medical Device Act"
という概念を学習させるに等しい。
これは若干無理があったと考えている。

最終的なプロンプトは以下の通り「この画像は(アメリカの)FFDCAに基づいて適切でしょうか?」に変更した。

> Is this image appropriate based on Federal Food, Drug, and Cosmetic Act?

コチラのプロンプトで学習するということは、
英語で事前学習を済ませたモデル・パラメーターに対して
FFDCAについての知識は一定量学習済みであることを仮定して、
それを日本の薬事法に適用できるように微調整する狙いがある。

このプロンプトの変更により、学習前の評価値が改善している。

(TODO: データを引用)

### 評価基準

評価基準は前述の通り、
問題を二項分類に落とし込んだことでわかりやすいものが利用できた。

その他の問題についても同様に、
機械的に評価できる形に落とし込むことが重要だと推定される。

### 学習のイテレーション

必ず成功する学習はないと考えるべし。
成功しやすい学習も、失敗しやすい学習もある。
ただし「もっとも良い学習結果」を得るには、
失敗しやすい学習を選ぶ必要がある場合もありそう。

結果、安定して良い学習を得るには、いずれにせよ回数をこなす必要がある。
よって1回の学習時間を短くし、
学習を自動で実行・評価できるワークフローを構築することが、
成功の必要条件となる。

### 学習方法

今回、学習はパラメーターを直接変更(バックプロパゲーション)する方式を用いた。
これはプログラムがシンプルで、学習のロジックを理解する上では優秀である。
しかし学習を繰り返すという観点からは好ましくはない。

その1つの理由がサイズの大きさで、
学習のたびにパラメーターを全て保存する必要があり、ディスクを大量に消費する。
これはLoRAなどの外付けアダプタ方式方式に変更することで、
小さくできる可能性がある。

もう1つの理由がバッチの小ささで、学習が発散して失敗しやすくなっている。
これはGravient Accumrationといった、
ワークエリア(VRAM)を小さいまま
バッチを大きくする技術を利用できる可能性がある。

## コードと実データ

### コード

実験に利用したコードの一部を以下に示す。
実行に必要な全体ではないのに加えパラメタライズもしてないので、
このままでは再現実験には使えない

* (TODO: モデルとのパラメーターをダウンロードするスクリプト)
* 学習+パラメーター保存: [learn0.py](./learn0.py)
* パラメーター評価: [model\_validate2.py](./model_validate2.py)
* 学習+評価(保存無し): [learn2+valid.py](./learn2+valid.py)

### データー

(TODO: データーを抜粋して掲載)

## 課題

アニメーションGIFは特定の(おそらく最終)フレームだけを学習・推論に利用している。
そのため強いノイズになってる可能性がある。
適切なフレームを利用するか、データから除外するかしたほうが良い。
