# Merlin Transformers4Rec tutorial

参考: [古い記録](./README.md)

Transformers4Rec は、ユーザーの行動に基づき何かしらのアイテムを提案するレコメンドシステムを、
LLMで一般的になった Transformer を用いて実現するためのソリューションである。
特に [Huggging Face Transformers][hf_transformers] で利用可能な Transformers モデルを任意に利用できる点が特徴である。
また入力にアイテムについてのデータだけでなく、
アイテムをカテゴライズする副情報を追加できる点に強みがある。

[hf_transformers]:https://huggingface.co/docs/transformers/ja/index

Transformers4Recのレポジトリには [実行可能な例][t4rec_examples] がいくつか収録されている。
いずれの例も概ね同じようなデータ処理フローを行っており、
その内容は以下のようになっている。

1. データを 抽出、変換、出力 (ETL) する。またその操作内容をWorkflowとして保存する
2. 学習してモデルを保存する
3. 1と2を [Triton Inference Server][triton] にロードし、提案システムとして稼働する

しかし実際に例を実行して確認したところ、
上記のステップのうち、ステップ3のTritonサーバーが上手く機能していないことがわかった。
どのような入力をしても出力が全て 0 になってしまっていた。

Tritonサーバーが機能しない理由は不明だが、
まずはとにかく提案システムが機能していることを確認するために [examples/tutorial][tutorial] をベースに、
提案結果を受け取れることを目標に試行錯誤し、実際に提案といえるものを受け取れるところまで確認した。

[t4rec_examples]:https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples
[triton]:https://github.com/triton-inference-server/server
[tutorial]:https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/tutorial

[examples/tutorial][tutorial] はデータ処理フローのうち
ステップ1のETLとステップ2の学習を行うようになっている。
また Transformers +副情報を使った方法だけでなく、伝統的な RNN (Recurrent Neural Network) や、副情報なしでの Transformers を使った方法との比較ができるようになっている。

それに対して以下のスクリプトを作成し、提案システムが機能することと、
その背景にある仕組みの一部を確認した。

* [`04-train_and_save.py`](./04-train_and_save.py)

    ステップ1で事前処理されたデータを使い、Transformers+副情報で学習し保存する。
    ステップ2に相当する。

* [`06-load.py`](./06-load.py)

    スクリプト04で保存したモデルを読み込む、ミニマルなサンプル。
    その後のオペレーションを構築する事前の確認用。

* [`07-eval-with-trainer.py`](./07-eval-with-trainer.py)

    スクリプト04で保存したモデルを読み込み、既知のデータに対して評価するスクリプト。
    モデルの学習と保存と読込が上手く行っているかを確認する目的で作成した。
    スクリプト04の学習時に表示される統計情報と概ね一致することが確認できている。

* [`08-load_workflow.py`](./08-load_workflow.py)

    ステップ1で事前処理した `Workflow` を読み込む、ミニマルなサンプル。
    生データ Oct-2019.parquet (ステップ1で作成した) を変換 (transform) し、
    意図したとおりに期待することを確認した。

* [`09-compose_workflow.py`](./09-compose_workflow.py)

    ステップ1で `Workflow` を作成するコードを切り出したもの。
    `Workflow` の処理内容を確認するためのもの。

* [`10-predict_with_trainer.py`](./10-predict_with_trainer.py)

    Trainer と検証用データを使って推論(predict)を実行するサンプルコード。

* [`11-predict_manual_etl.py`](./11-predict_manual_etl.py)

    人工的な4つのデータに対して「提案」を取得するサンプルコード。
    1つのデータは1セッションに相当し、
    1セッションに対して100個のアイテムが提案される。

## ポイント

*   初めての訪問者にも提案がされる
*   売れ筋の商品ほど提案されやすい
*   同じものを買ってる人でも、時間が異なると異なる提案がされている
*   同じ時間に別のモノを買ってると、異なる提案がされている

## 時間の特徴量の循環表現

時間的な特徴量は循環的な性質を持つため sin & cos のペアを使って表現している。
今回のケースでは曜日ごとに売れるものが特徴づけられているのではないか、
という仮定を置いている。

参考: https://ianlondon.github.io/posts/encoding-cyclical-features-24-hour-time/

## 価格の(対数)変換

価格情報は対数変換されている。その目的は:

* 価格のスケールを小さくする
* 分布を正規分布に近づける

また `nvt.opt.Normalize` で正規化することで
平均0、標準偏差1(の正規分布?)へと標準化している。

## `Workflow` の副作用

商品や初品カテゴリのデータは `Workflow` でIDがリナンバリングされている。
元々の値と `Workflow` が生成したIDとの対応は、
`workflow_etl/categories` 内の各 parquet ファイルに格納されている。

```console
$ ls -1 workflow_etl/categories/
cat_stats.category_id.parquet/
unique.brand.parquet
unique.category_code.parquet
unique.category_id.parquet
unique.event_type.parquet
unique.product_id.parquet
unique.user_id.parquet
```

ID生成時には要素の多い順にIDが順番に振られていた。
そのため提案結果内には、小さい番号≒頻出する要素が優先されやすい、
という傾向がある。
しかし場合によっては、大きい番号が先に出たりと、逆転することもある。

## まとめ

* Triton Inference Serverを使わずに推論(提案)が利用できた
* 提案内容はそれっぽいものになっていた
* アイテムに動的に追加変更がある場合、提案システムの工夫が必要

(一連の流れを確認できる Jupyter notebook を作った方が良いかも)
