# Merlin Transformers4Rec tutorial

参考: [古い記録](./README.md)

Transformers4Rec (t4rec) は、ユーザーの行動に基づき何かしらのアイテムを提案するレコメンドシステムを、
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

またコレのスクリプトをまとめて実行可能にするために [ノートブック][notebook] を作成した。

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

## 確認用のJupyter notebook (trial.ipynb) の使い方

[ノートブック][notebook] を利用するには以下の手順に従うこと

1. 本ディレクトリに移動する

        $ cd NVIDIA-Merlin/Transformers4Rec

2. notebookディレクトリをマウントしてdockerコンテナを起動する

        $ ./docker_t4rec_run -d notebook -n t4rec

3. dockerコンテナ内に kagglehub モジュールをインストールする

    本ステップが必要なのは学習用データ `2019-Oct.csv` のダウンロードが必要な時だけ

        # pip3 install -U kagglehub

4. dockerコンテナ内に Kaggle のAPIキーを設定する

    本ステップが必要なのは学習用データ `2019-Oct.csv` のダウンロードが必要な時だけ。
    APIキーの取得方法は [公式ドキュメント](https://www.kaggle.com/docs/api#authentication) を参照すること

        # export KAGGLE_USERNAME=xxxxxxxx
        # export KAGGLE_KEY=xxxxxxxx

5. dockerコンテナ内でjupyter notebookを起動する

        # cd / ; jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token=''

6. <http://127.0.0.1:8888/lab/tree/workspace/data> をブラウザで開き `trial.ipynb` を開く


## データ構造

t4recで扱うデータ構造(スキーマ)は複数存在する。

*   生CSV
*   CSVを Parquet に変換したもの
*   ETLによりセッション毎にまとめた Parquet
*   日付ごとに分割したもの
*   t4rec モデルの入力
*   t4rec モデルの出力

あるデータ構造は、その1つ前段階のデータ構造からの変換で作られている。

以下ではそれぞれのスキーマを説明する。

### 生CSV

ファイル名: `notebook/2019-Oct.csv`

カラム名        | 説明
----------------|--------------------------------------
`event_time`    | イベント発生時刻
`event_type`    | イベントの種別
`product_id`    | 対象となる製品ID
`category_id`   | 製品カテゴリのID
`category_code` | 製品カテゴリのコード
`brand`         | 製品のブランド名
`price`         | 価格
`user_id`       | イベントを発生させたユーザーID
`user_session`  | ユーザーのセッションID

### CSV を Parquet に変換したもの

基本的にCSVをParquetとして扱いやすくしたもの。

ファイル名: `notebook/Oct-2019.parquet`

カラム名                   | 説明
---------------------------|--------------------------------------
`event_type`               | イベント発生時刻
`product_id`               | 対象となる製品ID
`category_id`              | 製品カテゴリのID
`category_code`            | 製品カテゴリのコード
`brand`                    | 製品のブランド名
`price`                    | 価格
`user_id`                  | イベントを発生させたユーザーID
`event_time_ts`            | `event_time` の秒表記
`user_session`             | ユーザーのセッションID
`prod_first_event_time_ts` | その製品が初めて登場した時刻

サンプルコードではこの時点でデータを最初の7日間に絞り込んでいる。

### ETLによりセッション毎にまとめた Parquet

機械学習モデルに入力できる形に近づけたもの。

ファイル名: `notebook/processed_nvt/part_0.parquet`

カラム名                              | 説明
--------------------------------------|--------------------------------------
`user_session`                        | ユーザーのセッションID (正規化済)
`product_id-count`                    | 1セッションで参照された製品の数
`product_id-list`                     | 参照された製品のIDのリスト (最大20、IDは正規化済)
`category_code-list`                  | 参照された製品のカテゴリコードのリスト (最大20、コードは正規化済)
`brand-list`                          | 参照された製品のブランドのリスト (最大20、ブランド名は正規化済)
`category_id-list`                    | 参照された製品のカテゴリIDのリスト (最大20、IDは正規化済)
`et_dayofweek_sin-list`               | イベント発生曜日を循環構造に変換した正弦(sin)成分 (最大20)
`et_dayofweek_cos-list`               | イベント発生曜日を循環構造に変換した余弦(cos)成分 (最大20)
`price_log_norm-list`                 | 参照された製品の価格データを正規化したもののリスト (最大20)
`relative_price_to_avg_categ_id-list` | 製品価格をカテゴリの平均価格に対して相対化したもの (最大20)
`product_recency_days_log_norm-list`  | 参照された各製品の最新性(?)を表すリスト (最大20)
`day_index`                           | セッションの開始日を代表インデックスとしたもの(1始り)

参照された製品をセッション毎にまとめている。

正規化済とは、ID等の離散値を出現頻度順に並べて多い方から順に番号を振りなおしていることを言う。
逆変換に必要なデータは `notebook/categories/unique.*.parquet` に格納されている。

`et_dayofweek_sin-list` と `et_dayofweek_cos-list` については [時間の特徴量の循環表現](#時間の特徴量の循環表現) に記載の通り、
時間的な特徴量は循環的な性質を持つため sin & cos のペアを使って表現している。
今回のケースでは曜日ごとに売れるものが特徴づけられているのではないか、
という仮定を置いている。

参考: https://ianlondon.github.io/posts/encoding-cyclical-features-24-hour-time/

### 日付ごとに分割したもの

前述のファイル(ETLによりセッション毎にまとめた Parquet)を `day_index` の値で日付ごとにパーティショニングして、
かつ学習用、テスト用、検証用の3つのデータセットに分割したもの。
`notebook/sessions_by_day/{day_index}` ディレクトリそれぞれに `train.parquet`, `test.parquet`, `valid.parquet` として保存される。

### t4rec モデルの入力

カラム名                              | Transformers4Rec用タグ
--------------------------------------|--------------------------------------
`product_id-list`                     | ID, ITEM, CATEGORICAL, LIST
`brand-list`                          | CATEGORICAL, LIST
`category_id-list`                    | CATEGORICAL, LIST
`product_recency_days_log_norm-list`  | CONTINUOUS, LIST
`et_dayofweek_sin-list`               | CONTINUOUS, LIST
`et_dayofweek_cos-list`               | CONTINUOUS, LIST
`price_log_norm-list`                 | CONTINUOUS, LIST
`relative_price_to_avg_categ_id-list` | CONTINUOUS, LIST

あるセッションがこれまでに参照した製品のリストとそレに対応する(t4recのコンテキストにおける)各種カテゴリのリストから、
続けて推薦すべき製品リストを提案するという建て付けになっている。

タグの意味付けはおおむね以下のような印象:

* ITEM, ID - 推薦対象となるアイテムとその識別情報
* CATEGORICAL - 推薦のヒントとなるアイテムのカテゴリ情報
* CONTINUOUS - 推薦のヒントとなるアイテムの近接情報
* LIST - リストであることを付記する装飾タグ

詳細なデータは[ノートブック][notebook]の末尾に添付してある。

### t4rec モデルの出力

推薦アイテムのリストとそのlogitのリスト。
各リストは100個となっている。

カラム名         | 説明
-----------------|----------------------------------------------
`next-item``[0]` | 推薦する製品のIDリスト (100個)
`next-item``[1]` | 推薦する製品の logit (100個)

## まとめ

*   Triton Inference Serverを使わずに推論(提案)が利用できた
*   提案内容はそれっぽいものになっていた
*   アイテムに動的に追加変更がある場合、提案システムの工夫が必要

    (WorkflowによるIDの発番が必要なため)

*   一連の流れを俯瞰で確認できる1枚の [Jupyter notebook][notebook] を作った

[notebook]:./notebook/trial.ipynb
