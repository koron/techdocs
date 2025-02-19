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
環境は Windows 11 Pro + Core i9 9900K + NVIDIA RTX 4070 (VRAM 12GB) + MSYS2 shell + Docker Desktop 。
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

なおここに書いた手順を実際に[再現するための章](#再現)を用意してある。
実際に再現する際はそちらを参照のこと。

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
プロンプト修正後かつ学習前の評価データは次の通り。
Tp+Tnが微増したためaccuracyとprecisionは改善したが、
Tpが減ったためにrecallは微減。
F値は微々増といったところ。

```
# プロンプト修正後かつ学習前
Result: dataset/data_shuffle.jsonl
  tp=19 tn=41 fp=12 fn=24
  accuracy:  0.625
  precision: 0.6129032258064516
  recall:    0.4418604651162791
  F-measure: 0.5135135135135135

# (比較用)プロンプト修正前かつ学習前
Result:
  tp=20 tn=38 fp=15 fn=23
  accuracy:  0.6041666666666666
  precision: 0.5714285714285714
  recall:    0.46511627906976744
  F-measure: 0.5128205128205128
```

9:1に分割して学習して trained2.npz を得ての評価データは以下の通り。

```
# プロンプト修正後かつ9:1で学習後
Result: ./dataset/data_shuffle.jsonl
  tp=39 tn=52 fp=1 fn=4
  accuracy:  0.9479166666666666
  precision: 0.975
  recall:    0.9069767441860465
  F-measure: 0.9397590361445783
Result: ./dataset/data_train.jsonl
  tp=31 tn=48 fp=0 fn=0
  accuracy:  1.0
  precision: 1.0
  recall:    1.0
  F-measure: 1.0
Result: ./dataset/data_valid.jsonl
  tp=4 tn=4 fp=1 fn=1
  accuracy:  0.8
  precision: 0.8
  recall:    0.8
  F-measure: 0.8000000000000002

# (比較用)プロンプト修正前かつ9:1で学習後
Result: ./dataset/data_shuffle.jsonl
  tp=35 tn=51 fp=2 fn=8
  accuracy:  0.8958333333333334
  precision: 0.9459459459459459
  recall:    0.813953488372093
  F-measure: 0.875
Result: ./dataset/data_train.jsonl
  tp=31 tn=48 fp=0 fn=0
  accuracy:  1.0
  precision: 1.0
  recall:    1.0
  F-measure: 1.0
Result: ./dataset/data_valid.jsonl
  tp=3 tn=3 fp=2 fn=2
  accuracy:  0.6
  precision: 0.6
  recall:    0.6
  F-measure: 0.6
```

いずれもプロンプト修正後のほうが若干ながら評価が良い。

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


## ディレクトリとファイル

* [bin/](./bin) Docker環境を整えるためのスクリプト

    ホスト(ローカル側)で実行することを想定している。
    Linuxや、Windows(+Docker Desktop)のMSYS2環境で、実行することを想定している。

    * [bin/docker-build.sh](./bin/docker-build.sh) 実験環境のコンテナイメージを作成するためのスクリプト
    * [bin/docker-run.sh](./bin/docker-run.sh) 実行環境のコンテナを開始するためのスクリプト。必要なボリュームマウント等を行っている

* [cache/](./cache) コンテナ内でダウンロードしたモデルなどを保存・永続化しておく場所。dockerのvolumeでも良いのだが、今回はマウントした。
* [Dockerfile](./Dockerfile) コンテナイメージの定義
* [playground/](./playground) コンテナ内でのワークディレクトリ

    データファイルや実行スクリプトを置く

    * [playground/dataset/](./playground/dataset/) 画像ファイルおよびそのメタデータ(JSONL)を置く
    * playground/dataset/data.jsonl 入力画像のメタデータなJSONL
        
        内容は以下のようになる。

        ```jsonl
        {"prefix":"","suffix":"no","image":"images/NG_image101.jpg"}
        {"prefix":"","suffix":"yes","image":"images/OK_image102.jpg"}
        ```

        * `prefix` 要素は空。
        * `suffix` 要素は想定される結果。
        * `image` 要素は `playground/dataset` からの相対指定で、画像ファイルのパス。

    * playground/dataset/images/ 画像ファイルの置き場
    * [playground/checkpoints/](./playground/checkpoints) 事後学習済みのモデルの置き場

    以下は実行スクリプト。Dockerコンテナ内で実行する必要がある

    * [playground/01-download_models.py](./playground/01-download_models.py) Paligemmnaのモデルとトークナイザーのモデルをダウンロードする。

        予めkaggleのアカウント情報をローカルの [~/.kaggle/kaggle.json](https://www.kaggle.com/docs/api#authentication) に保存しておく必要がある。

    * [playground/02-shuffle_data.sh](./playground/02-shuffle_data.sh) dataset/data.jsonl をシャッフルし学習用と検証用のセットに分け、1回の学習に利用できる形に成形する。

    * [playground/03-do_learn.py](./playground/03-do_learn.py) 学習を実行し、結果を評価する。オプションで学習後のモデルを保存できる。

    * [playground/04-validate.py](./playground/04-validate.py) 1つのモデルに対して、複数のデータセットを1度に検証できる。

## 再現

このドキュメントに書いた内容は以下の手順で再現できる。

### 事前準備

* GPUにアクセスできる形でdockerdを実行し、アクセスできるユーザーを用意する。Windowsの場合はDocker Desktopで良い。
* 本ディレクトリをコピーする
* playground/dataset にサンプルデータ data.jsonl と画像ファイルを配置する

### 再現手順

1. 実験用のDockerイメージを作成する(1回のみ)

    ```console
    $ ./bin/docker-build.sh
    ```

    初回は巨大(約10GB)なベースイメージをダウンロードするので、従量制ネットワークでやるべきではない。

    簡単なシェルスクリプトなのでWindowsのコマンドラインに読み替えて実行しても良い。

2. Dockerコンテナを実行する

    ```console
    $ ./bin/docker-run.sh
    ```

    以後はDockerコンテナ内での操作になる。
    少し複雑に見えるけれども、どのような設定でコンテナを実行しているかは読み取れるだろう。

3. モデルをダウンロードする(1回のみ)

    ```console
    $ ./01-download_models.py
    ```

4. 学習に使うデータセットを作成する(必要であれば複数回)

    ```console
    # 学習に使うデータセットを作成し dataset/set1/ に保存する。検証用データは10個
    $ ./02-shuffle_data.sh -v 10 -n set1

    # 学習に使うデータセットを作成し dataset/set2/ に保存する。検証用データは10個
    $ ./02-shuffle_data.sh -v 10 -n set2

    # 学習に使うデータセットを作成し dataset/set3/ に保存する。検証用データは20個
    $ ./02-shuffle_data.sh -v 20 -n set3
    ```

5. 学習を実行する

    ```console
    # dataset/set1 を用いて学習し ./checkpoints/trained-set1.npz に保存する
    $ ./03-do_learn.py -d set1 -w trained-set1

    # dataset/set2 を用いて2回学習し ./checkpoints/trained-set2.npz に保存する
    $ ./03-do_learn.py -d set2 -w trained-set2 -2
    ```

    学習の途中で出力の `loss` が1を超えて大きくなる場合、発散している疑いが濃厚なので中断し、再度学習をしたほうが良い。
    以下は学習が成功している際の出力例:

    ```
    step: 32/256   lr: 0.02995   loss: 0.3417
    step: 64/256   lr: 0.02804   loss: 0.3162
    step: 96/256   lr: 0.02370   loss: 0.2927
    step: 128/256   lr: 0.01774   loss: 0.2114
    step: 160/256   lr: 0.01127   loss: 0.1467
    step: 192/256   lr: 0.00549   loss: 0.4221
    step: 224/256   lr: 0.00149   loss: 0.0181
    step: 256/256   lr: 0.00000   loss: 0.0043
    ```

    以下は発散が疑われる失敗例であるので、中断して再学習めるべき。

    ```
    step: 32/256   lr: 0.02995   loss: 0.1789
    step: 64/256   lr: 0.02804   loss: 1.4870
    step: 96/256   lr: 0.02370   loss: 1.4092
    step: 128/256   lr: 0.01774   loss: 1.1224
    step: 160/256   lr: 0.01127   loss: 1.1452
    step: 192/256   lr: 0.00549   loss: 1.1013
    step: 224/256   lr: 0.00149   loss: 1.2169
    ```

    学習用データに対して各種評価値が1.0であれば、学習は収束したと言える。
    以下はその出力例:

    ```
    Result: ./dataset/set1/train.jsonl (1st trained)
      Tp=38 Tn=48 Fp=0 Fn=0
      accuracy:  1.0
      precision: 1.0
      recall:    1.0
      F-measure: 1.0
    ```

    一方でこれらの値が 1.0 に満たない場合は、収束に至ってないと考えられる。

6. モデルを評価する

    学習時に各評価値は計算されるが、以下のスクリプトを使うことで
    学習を伴わずに再度計算できる。

    ```console
    # set1 で学習したモデルを set2 のデータで評価する
    $ ./04-validate.py -m ./checkpoints/trained-set1.npz ./dataset/set2/train.jsonl ./dataset/set2/valid.jsonl
    Result: dataset/set2/train.jsonl
      Tp=34 Tn=48 Fp=1 Fn=3
      accuracy:  0.9534883720930233
      precision: 0.9714285714285714
      recall:    0.918918918918919
      F-measure: 0.9444444444444445
    Result: ./dataset/set2/valid.jsonl
      Tp=6 Tn=4 Fp=0 Fn=0
      accuracy:  1.0
      precision: 1.0
      recall:    1.0
      F-measure: 1.0
    ```

## 追試

### 「明らかにセーフ」なデータセットに関する検証

自前のフォトから明らかにセーフな普通の写真25枚を取り出し、学習前後のモデルに対して検証してみる。

学習前

```console
./04-validate.py ./dataset/safe.jsonl
Result: ./dataset/safe.jsonl
  Tp=15 Tn=0 Fp=0 Fn=10
  accuracy:  0.6
  precision: 1.0
  recall:    0.6
  F-measure: 0.7499999999999999
```

学習後

```console
./04-validate.py -m checkpoints/trained1.npz ./dataset/safe.jsonl
Result: ./dataset/safe.jsonl
  Tp=6 Tn=0 Fp=0 Fn=19
  accuracy:  0.24
  precision: 1.0
  recall:    0.24
  F-measure: 0.3870967741935484
```

もともと「良くはない」ものが、学習後には「明らかに悪い」になってる。

以上のことから学習に利用したデータセットの内容から強く影響を受けているので、
人の目で判断しにくいギリギリのケースを仕分けることにのみ利用可能な状態にあるのでは
と推測できる。

## 課題

アニメーションGIFは特定の(おそらく最終)フレームだけを学習・推論に利用している。
そのため強いノイズになってる可能性がある。
適切なフレームを利用するか、データから除外するかしたほうが良い。

データセットが偏っているのでは。
今回学習・検証に利用したデータセットは「ギリギリセーフ」と「ギリギリアウト」のデータセットになっており
「明らかにセーフ」や「明らかにアウト」なデータではない。
またこのデータセット学習後のモデルで「明らかにセーフ」や「明らかにアウト」のケースをinferenceしたら、
過学習により汎化能力を失ってることを考えると、ガタガタになる可能性が高い。
たしかめてみる。

## 補題

### 行指向入力データをシャッフルして分割する別解

```console
shuf indata.tsv | spilt -l 1000 --additional-suffix .tsv - outdir/splitted-
```

1万行のデータを上記コマンドで処理すると、シャッフルした上で
outdir/splitted-aa.tsv から outdir/splitted-aj.tsv まで
10個のファイルに分けられる。

### 学習時のVRAM

学習のスクリプトに以下のコードを追加することで、
VRAMの割り当てを都度割り当てにし、10個程度までバッチサイズを大きくできた。

```python
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
```

どうやらJAXはPythonのオブジェクトをVRAMに転送する際、
予めVRAMを確保したり、
そのオブジェクトが使われなくなってもキャッシュして可能な限り解放しないらしい。
結果、その後の計算に不要なメモリを持ち続け、必要なメモリを確保できないとのこと。

上記の設定はそれを必要に応じて確保するようにし、使わないVRAMは即時解放するようにしている。
結果、VRAMへのロードは増えオーバーヘッドがあるものの、必要なオブジェクトのみがVRAMに乗るため、
限られたVRAMサイズでも大きめのバッチで学習できるようになった。

### 学習の評価

分けたファイルの1つaaから先頭64個を抽出し、
バッチ10個 learning rate 0.001 で学習しパラメーターを保存。

その保存したパラメーターで分けたファイルの1つabを推定し、
学習前後の成績確認。

```console
$ ./14-validate_outtsv.py tmp/*-ab.tsv
Aggregate: tmp/out64-b10-ab.tsv
  Tp=854 Tn=17 Fp=155 Fn=13
  accuracy:  0.8383060635226179
  precision: 0.846382556987116
  recall:    0.9850057670126874
  F-measure: 0.9104477611940298
Aggregate: tmp/splitted-ab.tsv
  Tp=652 Tn=57 Fp=115 Fn=215
  accuracy:  0.6823869104908566
  precision: 0.8500651890482399
  recall:    0.7520184544405998
  F-measure: 0.7980416156670748
```

一見良くなっているように見えるがTnとFnが減りTpとFpが増えている。
これはnoだったものが学習でyesになったことを示しており
もともと正当がyesである率が高いため、
学習が成功したとは評価しづらい。
そのことはprecisionが下がっていることが裏付けている。

学習に使うデータを再検討したほうが良かろう。

aaセットから `false positive/negative` だけ、64件を取り出して学習してみる。

```console
$ ./16-filter_train_data.py tmp/splitted-aa.tsv | head -64 > tmp/false64.tsv
$ ./15-train.py -d tmp/false64.tsv -i dataset/pub_judges -b 10 -r 0.001 -s checkpoints/false64-r0.001-b10.npz

$ ./13-inference.py -l ./checkpoints/false64-r0.001-b10.npz -d tmp/splitted-ab.tsv -i dataset/pub_judges -b 4 | tee tmp/out-false64-r0.001-b10-ab.tsv

$ ./13-inference.py -l ./checkpoints/false64-r0.001-b10.npz -d tmp/splitted-aa.tsv -i dataset/pub_judges -b 4 | tee tmp/out-false64-r0.001-b10-aa.tsv

$ ./14-validate_outtsv.py tmp/out-false64-r0.001-b10-aa.tsv tmp/out-false64-r0.001-b10-ab.tsv tmp/splitted-ab.tsv
Aggregate: tmp/out-false64-r0.001-b10-aa.tsv
  Tp=880 Tn=0 Fp=159 Fn=0
  accuracy:  0.8469682386910491
  precision: 0.8469682386910491
  recall:    1.0
  F-measure: 0.9171443460135488
Aggregate: tmp/out-false64-r0.001-b10-ab.tsv
  Tp=867 Tn=0 Fp=172 Fn=0
  accuracy:  0.8344562078922041
  precision: 0.8344562078922041
  recall:    1.0
  F-measure: 0.9097586568730325
Aggregate: tmp/splitted-ab.tsv
  Tp=652 Tn=57 Fp=115 Fn=215
  accuracy:  0.6823869104908566
  precision: 0.8500651890482399
  recall:    0.7520184544405998
  F-measure: 0.7980416156670748
```

yes とだけ返すように学習してしまったようだ。
false negative のデータが多いため (Fn=215)
yes と返すことを学習するのではと予想できる。

true positive だけを除いた64件のデータで8バッチにして8回(計64ステップ)学習してみる。

```console
$ grep -v '\bTrue\b.*\byes$' ./tmp/splitted-aa.tsv | head -64 > tmp/noTp64.tsv

$ ./15-train.py -d tmp/noTp64.tsv -i dataset/pub_judges -t 8 -b 8 -r 0.001 -s checkpoints/noTp64-t8-r0.001-b80.npz

$ ./13-inference.py -l ./checkpoints/noTp64-t8-r0.001-b8.npz -d tmp/splitted-ab.tsv -i dataset/pub_judges | tee tmp/out-noTp64-t8-r0.001-b8-ab.tsv
$ ./13-inference.py -l ./checkpoints/noTp64-t8-r0.001-b8.npz -d tmp/splitted-aa.tsv -i dataset/pub_judges | tee tmp/out-noTp64-t8-r0.001-b8-aa.tsv
$ ./13-inference.py -l ./checkpoints/noTp64-t8-r0.001-b8.npz -d tmp/noTp64.tsv      -i dataset/pub_judges | tee tmp/out-noTp64-t8-r0.001-b8-xx.tsv

$ ./14-validate_outtsv.py tmp/out-noTp64-t8-r0.001-b8-xx.tsv tmp/noTp64.tsv
Aggregate: tmp/out-noTp64-t8-r0.001-b8-xx.tsv
  Tp=41 Tn=23 Fp=0 Fn=0
  accuracy:  1.0
  precision: 1.0
  recall:    1.0
  F-measure: 1.0
Aggregate: tmp/noTp64.tsv
  Tp=0 Tn=5 Fp=18 Fn=41
  accuracy:  0.078125
  precision: 0.0
  recall:    0.0

$ ./14-validate_outtsv.py tmp/out-noTp64-t8-r0.001-b8-ab.tsv tmp/splitted-ab.tsv
Aggregate: tmp/out-noTp64-t8-r0.001-b8-ab.tsv
  Tp=813 Tn=41 Fp=131 Fn=54
  accuracy:  0.821944177093359
  precision: 0.861228813559322
  recall:    0.9377162629757786
  F-measure: 0.8978464936499171
Aggregate: tmp/splitted-ab.tsv
  Tp=652 Tn=57 Fp=115 Fn=215
  accuracy:  0.6823869104908566
  precision: 0.8500651890482399
  recall:    0.7520184544405998
  F-measure: 0.7980416156670748

$ ./14-validate_outtsv.py tmp/out-noTp64-t8-r0.001-b8-aa.tsv tmp/splitted-aa.tsv
Aggregate: tmp/out-noTp64-t8-r0.001-b8-aa.tsv
  Tp=837 Tn=47 Fp=112 Fn=43
  accuracy:  0.8508180943214629
  precision: 0.8819810326659642
  recall:    0.9511363636363637
  F-measure: 0.9152542372881356
Aggregate: tmp/splitted-aa.tsv
  Tp=648 Tn=52 Fp=107 Fn=232
  accuracy:  0.6737247353224254
  precision: 0.8582781456953642
  recall:    0.7363636363636363
  F-measure: 0.7926605504587156
```

true positive を除外しない aaからの64件のデータで、8バッチに8回(計64ステップ)学習し、傾向をみる。

```console
$ ./15-train.py -d tmp/test64.tsv -i dataset/pub_judges -b 8 -r 0.001 -t8 -s checkpoints/test64-t8-r0.001-b8.npz

$ ./13-inference.py -l checkpoints/test64-t8-r0.001-b8.npz -d tmp/splitted-ab.tsv -i dataset/pub_judges | tee tmp/out-test64-t8-r0.001-b8-ab.tsv
$ ./14-validate_outtsv.py tmp/out-test64-t8-r0.001-b8-ab.tsv ./tmp/splitted-ab.tsv
Aggregate: tmp/out-test64-t8-r0.001-b8-ab.tsv
  Tp=866 Tn=20 Fp=152 Fn=1
  accuracy:  0.8527430221366699
  precision: 0.8506876227897839
  recall:    0.9988465974625144
  F-measure: 0.9188328912466843
Aggregate: ./tmp/splitted-ab.tsv
  Tp=652 Tn=57 Fp=115 Fn=215
  accuracy:  0.6823869104908566
  precision: 0.8500651890482399
  recall:    0.7520184544405998
  F-measure: 0.7980416156670748

./13-inference.py -l checkpoints/test64-t8-r0.001-b8.npz -d tmp/splitted-aa.tsv -i dataset/pub_judges | tee tmp/out-test64-t8-r0.001-b8-aa.tsv
./14-validate_outtsv.py tmp/out-test64-t8-r0.001-b8-aa.tsv ./tmp/splitted-aa.tsv
Aggregate: tmp/out-test64-t8-r0.001-b8-aa.tsv
  Tp=878 Tn=25 Fp=134 Fn=2
  accuracy:  0.8691049085659288
  precision: 0.8675889328063241
  recall:    0.9977272727272727
  F-measure: 0.9281183932346723
Aggregate: ./tmp/splitted-aa.tsv
  Tp=648 Tn=52 Fp=107 Fn=232
  accuracy:  0.6737247353224254
  precision: 0.8582781456953642
  recall:    0.7363636363636363
  F-measure: 0.7926605504587156
```

#### ここまでのまとめ (2025-02-19)

* `yes` と答える圧力が強い (全体の85%以上が `yes` であることによる)
* 学習はイテレーションをした方が良い。過学習(常にyesと答える)が抑制される傾向にある
* 学習データからtrue positiveを除いた方が `yes` 圧力は弱まる
* 「true negative が減ってしまう」&「false positive が増えてしまう」ことが問題
    * true negative と false positive のデータだけで学習させたら、どうなるか見たほうが良い
