# sqlite-vss 入門

本文章では、sqlite-vssの入門的な使い方と、関連した知識を解説します。
現在ドラフト相当の段階のため、後に記述が変更・追加になる場合があります。

<https://github.com/asg017/sqlite-vss>

sqlite-vss とは、SQLiteでFaiss(ベクトル検索ライブラリ)を使えるようにする、SQLiteの拡張モジュールです。
SQLiteは言わずとも知れた組込み用のRDBMSで、多くの言語にバインディング・ドライバーが存在するとてもメジャーなRDBMSの1つです。
もう一方のFaissはFacebookが開発・公開している、密なベクトルの効率的な類似性検索とクラスタリングのためのライブラリです。
平たく言うと、sqlite-vssはSQLiteに大規模なベクトルの検索機能を追加してくれる拡張モジュール、というわけです。

SQLiteでベクトル検索ができると何が良いのでしょう?
イメージしやすいように敢えて単純化すると、ベクトル検索により画像検索ができるようになります。
画像データは巨大な高次元のベクトルと見做せます。
探したい画像のベクトルを指定できさえすれば、類似するベクトルを探し出すことで画像検索が実現できます。
全文検索にも近いことがいえます。
探している文章のベクトルを指定できれば、ベクトル検索にて求めている求めている文章を見つけ出せるという仕組みです。
画像や全文など、検索アルゴリズムの実装が容易ではなさそうなものをベクトルと捉える・ベクトルに変換するだけで、一律に検索できるようにしてしまうのがベクトル検索というわけです。

ベクトル検索のアルゴリズム自体は実はとても簡単です。
探し出したいベクトルに対して、検索対象となる全ベクトルとの距離を求めて、最も近いものから順番に必要な数を列挙するだけです。
しかし現実に必要になるベクトル検索は、検索対象の数がとても多いのに加えて、ベクトルの次元がとても多いのです。
検索対象の数は億(10^8)に、ベクトルの次元は万(10^4)に、それぞれ届くかそれ以上になりえます。
こうなると素朴なベクトル検索アルゴリズムは `O(m*n)` の計算オーダーになりますから、文字通り天文学的な時間がかかる、すなわち遅くなることが容易に想像できます。

ですから現実のベクトル検索では、素朴なアルゴリズムではなく、空間や時間的に効率を追求したアルゴリズムが考案され使われています。
Faissはそのようなベクトル検索アルゴリズムをまとめてライブラリ化したものです。
sqlite-vssによりSQLiteからFaissが提供するベクトル類似性検索アルゴリズムにできるようになるわけです。

ちなみにベクトル類似性検索のことを近似最近傍探索とも言います。
そちらに興味がある場合は以下のリンクのスライドが詳しいのでとても参考になります。

[近似最近傍探索の最前線](https://speakerdeck.com/matsui_528/jin-si-zui-jin-bang-tan-suo-falsezui-qian-xian)

sqlite-vssの使い方を見ていきましょう。
本記事の執筆時点でリリースされているsqlite-vssのバージョンはv0.1.0です。
筆者はWSL2上のUbuntuで実際に試して動作を確認しました。
以下ではLinux (64bit)を前提として説明していきます。

下記のリンク先でLinuxやmacOS用のコンパイル済みバイナリが入手できます。

<https://github.com/asg017/sqlite-vss/releases/tag/v0.1.0>

以下の2つのtar.gzファイルをダウンロードしましょう。

* [sqlite-vss-v0.1.0-vector0-linux-x86\_64.tar.gz](https://github.com/asg017/sqlite-vss/releases/download/v0.1.0/sqlite-vss-v0.1.0-vector0-linux-x86_64.tar.gz)
* [sqlite-vss-v0.1.0-vss0-linux-x86\_64.tar.gz](https://github.com/asg017/sqlite-vss/releases/download/v0.1.0/sqlite-vss-v0.1.0-vss0-linux-x86_64.tar.gz)

sqlite-vssを使うにはsqlite-vssとは別にsqlite-vectorというSQLite拡張モジュールが必要です。
1つ目のファイルがsqlite-vectorのものです。
sqlite-vectorは、SQLiteにFaissで利用可能なベクトル型を提供します。
また、そのベクトル型をJSONやBLOBなど、別の形式と相互変換する関数も提供します。

ダウンロードしたtar.gzファイルを展開するとvector0.soとvss0.soが得られます。
これらの拡張モジュールをSQLiteに読み込むには、両ファイルが存在するディレクトリをカレントとしてsqlite3を起動した上で、以下の命令を実行します。
もちろん別の場所に置いておき、相対パスや絶対パスで読み込むこともできます。

```sql
.load ./vector0
.load ./vss0
```

ここから先に進む前にsqlite-vssの動作の全体像を概観しておきましょう。
sqlite-vssではベクトル検索を利用するために、仮想テーブルをまるでインデックスのように用います。
検索対象となる本体テーブルと共通のrowidを仮想テーブルに持たせ、仮想テーブル側でベクトル検索をしてrowidを確定したのちに、本体テーブルの必要なカラムを参照するという流れになります。

まず以下のように本体テーブルを作成します。
今回は単語≒文字列に300次元のベクトルを与えてベクトル検索を試します。
labelカラムには単語を示す文字列を保持します。
vectorカラムには生ベクトルを保持します。
生ベクトルは、仮想テーブルに転記してしまえば原理的には必要なくなるのですが、現時点では消せないためそのまま保持し続けます。

```sql
CREATE TABLE word (
    label  TEXT,
    vector BLOB
);
```

入力データは下記のようなフォーマットのTSVファイルになっています。
フォーマットを書き下すと `{単語}{タブ文字}{JSON配列で表現された300次元のベクトル}` となります。
実際に投入するデータの入手方法と整形方法は別途説明します。

```
We      [-0.0821,-0.1611,0.2441, ... ,0.5656,-0.0609,0.3025]
than    [-0.0524,0.1035,0.0511,  ... ,-0.0188,0.0140,0.0387]
only    [-0.0379,0.0619,-0.0718, ... ,0.0768,-0.0735,0.0020]
```

入力データのファイル名を`2e3.tsv`とした場合、その読み込みコマンドは以下のようになります。

```sql
-- .importでTSVフォーマットを読み込むように指定
.mode tab

-- ファイルをwordテーブルへ読み込み
.import 2e3.tsv word

-- vectorカラムをJSONテキストから、ベクトル型のBLOB値に変換することで、容量を削減
UPDATE word SET vector = vector_to_blob(vector_from_json(vector));

-- 前段で実行した容量削減をディスクに反映
VACUUM;
```

後半はベクトルの保存サイズを圧縮しています。
TSVを`.import`した直後はベクトルがJSONとして保存されるので、1ベクトルにつき約2200バイトが用いられます。
これをsqlite-vectorが提供するベクトルBLOBに変換することで約1200バイトに削減できます。
変換はJSONからベクトル型に変換したのちに、ベクトルのBLOB表現に再度変換しています。
なおベクトル型はそのままではテーブルに保存できないので、ベクトルのBLOB表現に変換する必要があります。

次に、もっともシンプルなベクトル検索用の仮想テーブルを作成します。
以下にそのSQL文を示します。
この仮想テーブルでは前述の素朴なベクトル検索アルゴリズムが実施されます。
裏を返すとsqlite-vssでは、仮想テーブルの作り方で検索アルゴリズムを選択する、ということでもあります。

```sql
CREATE VIRTUAL TABLE vss_word USING vss0 (
    vector(300)
);
```

この仮想テーブルにデータを登録するために、以下のSQL文を実行しwordテーブルからベクトルを転記します。
rowidも合わせて転記することで、インデックスとして機能するようにしています。
また利用するアルゴリズム次第では、転記の前にトレーニング・事前最適化をする必要がありますが、それは後に説明します。

```sql
INSERT INTO vss_word(rowid, vector)
    SELECT rowid, vector FROM word;
```

ここまでできればあとは検索するだけです。
以下のSQLを実行してみましょう。
このSQLでは`food`に近いベクトルの単語を、近い方から10個、その単語と距離を合わせて表示します。
6行目の`'food'`を変更すれば別の単語で検索できます。
ただしベクトルを指定しての検索であるため、wordテーブルに存在する単語でしか検索できないことに留意が必要です。

```sql
SELECT w.label, v.distance FROM vss_word AS v
  JOIN word AS w ON w.rowid = v.rowid
  WHERE vss_search(
    v.vector,
    vss_search_params(
      (select vector from word where label = 'food'),
      10
    )
  );
```

この書き方は、Ubuntuの標準パッケージに含まれるSQLiteが3.37であるため、`vss_search_params()`を使う必要があり若干冗長になっています。
3.41.0以降のSQLiteを使う場合には、sqlite-vssのドキュメントにあるような、より短い形式で書けるはずです。
以下にはその形式で書き直したものを示します。
こちらのほうが上のモノよりは読みやすいでしょう。

```sql
SELECT w.label, v.distance FROM vss_word AS v
  JOIN word AS w ON w.rowid = v.rowid
  WHERE vss_search(
    v.vector,
    (select vector from word where label = 'food')
  )
  LIMIT 10;
```

検索方法を概観すると次のようになっています。

1. 検索する単語`food`のベクトルを決定する
2. `vss_search_params()`関数を用いて1のベクトルと必要な件数から、検索パラメータを作成する
3. WHERE句に`vss_search()`関数を指定して、仮想テーブルを検索する。同関数には、仮想テーブル内のカラムと、2の検索パラメーターを指定する
4. 仮想テーブルのrowidと本体テーブルのrowidを突き合わせて単語文字列を取得し、距離とともに表示する

この検索方法の大枠はベクトル検索アルゴリズムを変更しても変わりません。
SQLiteとsqlite-vssにより検索方法が抽象化されていると言えるでしょう。
仮想テーブル`vss_word`の`vector`カラムは、Faissのインデックスに相当しています。
sqlite-vssでは1つの仮想テーブルに複数のカラム、すなわちFaissのインデックスを追加することも可能です。
Faissのインデックスが複数の検索アルゴリズムに対する抽象インターフェースを提供しているため、sqlite-vssによる検索が抽象化される、と表現するほうがより正確でしょう。

ここまで来ると「どのように仮想テーブルにカラムを作るか」が重要となります。
それは「どの検索アルゴリズムを選択するか」ということであり、言い換えれば「どのようなパラメーターでFaissのインデックスを作るか」ということです。
具体例を以下に示します。

```sql
CREATE VIRTUAL TABLE vss_hnsw_word USING vss0 (
    vector(300) factory="HNSW,Flat,IDMap2"
);
```

前述の`vss_word`と比べて`vector`カラムに`factory="HNSW,Flat,IDMap"`という指定が増えていることが一目瞭然です。
この`factory`に指定した文字列でFaissのインデックスを作る際のパラメーターを指定しています。
この例で指定しているのは`HNSW,Flat,IDMap2`です。
この文字列はsqlite-vssからFaissの`faiss::index_factory()`に直接渡され、インデックスを作成するのに使われます。
Faissのindex factoryについてのドキュメントは下記のリンクを参照してください。

<https://github.com/facebookresearch/faiss/wiki/The-index-factory>

これよりも詳細な仕様を知る必要がある場合には`Faiss::index_factory()`のソースを見てください。
以下のリンクで参照できます。

<https://github.com/facebookresearch/faiss/blob/868e17f29493075742170885f1f57c7b9e61d9ea/faiss/index_factory.cpp#L883>

デフォルトのFaissインデックスの指定は`Flat,IDMap2`です。
上の`vss_hnsw_word`テーブルでは、これに加え`HNSW`が増えていることがわかります。
これらの指定の一つ一つの要素を分解して説明すると次の通りです。

* `IDMap2` ほぼ必須。Faissインデックス≒仮想テーブルに`rowid`カラムが追加され、保存できるようになる
* `Flat` ベクトルをそのまま記録し圧縮しない。圧縮するには`PQ{n}`等を指定する
* `HNSW` 検索にグラフベースの[Hierarchical Navigable Small World (HNSW)アルゴリズム](https://suzuzusu.hatenablog.com/entry/2020/12/14/020000)を利用する

つまり前述の`vss_hnsw_word`仮想テーブルは、検索にHNSWアルゴリズムを用いる未圧縮のFaissインデックスとなります。
この`vss_hnsw_word`仮想テーブルへのデータ登録には、事前トレーニングが必要となります。
仮想テーブルへベクトルを書き込む際に、一緒に疑似カラム`operation`へ固定値`'training'`を書き込むことでトレーニングが実行できます。
次のSQLはそのトレーニング方法を示したものです。

```sql
INSERT INTO vss_hnsw_word(operation, vector)
    SELECT 'training', vector FROM word;
```

トレーニングの実行の際に、採用しているアルゴリズムによって、必要な件数やサイズ次第で警告がでることがありますが、今は無視してください。
またトレーニングにかかる時間もアルゴリズムと件数によって大きく変わってきます。

トレーニングが終わった後は実際にベクトルデータを登録・転記します。
以下はそのSQLですが、前述の`vss_word`に登録したときとほぼ同じで、仮想テーブル名が変わっただけです。

```sql
INSERT INTO vss_hnsw_word(rowid, vector)
    SELECT rowid, vector FROM word;
```

検索方法も全く同じで、利用する仮想テーブルが変わるだけです。
以下にそのSQLを示します。

```sql
SELECT w.label, v.distance FROM vss_hnsw_word AS v
  JOIN word AS w ON w.rowid = v.rowid
  WHERE vss_search(
    v.vector,
    vss_search_params(
      (select vector from word where label = 'food'),
      10
    )
  );
```

ただし検索結果は、素朴なアルゴリズムではなくHNSWアルゴリズムを利用していますので、異なってきます。
(TODO: その実際の例を提示する予定です)

最後にベクトルのエンコード方法を変更して、実質的な圧縮をしてみましょう。
ここまでベクトルのエンコードは`Flat`であり、1次元あたり32ビット=8バイトを要していました。
これは決して小さくなく、300次元のベクトル1つを記録するのに1200バイトかかる計算になります。
1つあたり約10^3バイトかかるベクトルが億(10^9)の単位あると、それだけで10^12つまりテラバイトの記憶領域を必要とします。
これは現実的ではありません。

そこで[Product Quantization (PQ)](https://www.google.com/search?q=%E7%9B%B4%E7%A9%8D%E9%87%8F%E5%AD%90%E5%8C%96&oq=%E7%9B%B4%E7%A9%8D%E9%87%8F%E5%AD%90%E5%8C%96&ie=UTF-8)を用いて、ベクトルを圧縮してみます。
そのような仮想テーブル≒Faissインデックスを作るのが以下のSQLです。

```sql
CREATE VIRTUAL TABLE vss_hnsw_pq_word using vss0 (
    vector(300) factory="HNSW,PQ15,IDMap2"
);
```

`vss_hnsw_word`仮想テーブルと比べて、`factory`の`Flat`が`PQ15`に置き換わっています。
`PQ`の後ろの数字はベクトルを幾つに分割するか(もしくは幾つ毎に分割するか: 要確認)のパラメーターで、次元数を割り切れる必要があります。
ここでは15を指定したので、次元数300を割り切れることが自明です。

事前学習の方法は`vss_hnsw_word`仮想テーブルの時と全く同じです。
ただし学習には、PQを指定したことで、より時間がかかることに注意してください。
以下のSQLがその学習を行うものです。

```sql
INSERT INTO vss_hnsw_pq_word(operation, vector)
    SELECT 'training', vector FROM word;
```

当然データの投入も同じです。
以下のSQLはデータの投入のためのものです。

```sql
INSERT INTO vss_hnsw_pq_word(rowid, vector)
    SELECT rowid, vector FROM word;
```

もちろん検索方法も同じです。
ただしPQによりベクトルを圧縮・簡略化してしまったために、ベクトル間の距離関係が変わってしまい、結果の順序や内容が変わります。
(TODO: その実際の例を提示する予定です)
以下は検索のためのSQLです。

```sql
SELECT w.label, v.distance FROM vss_hnsw_pq_word AS v
  JOIN word AS w ON w.rowid = v.rowid
  WHERE vss_search(
    v.vector,
    vss_search_params(
      (select vector from word where label = 'food'),
      10
    )
  );
```

以上、sqlite-vssの入門的な使い方と、関連した知識の解説でした。

## データの入手・加工方法

本文で利用したデータは下記のリンクから入手できます。

* [2e3.tsv](https://raw.githubusercontent.com/koron/techdocs/main/sqlite-vss-getting-started/2e3.tsv) 2000個のベクトルが入ったデータ(約4.4MB)

### オリジナルデータの入手・加工方法

前述のデータのオリジナルデータは以下のリンクから入手できます。
圧縮して約1.3GB、展開して4.3GBという大きなデータなので、ディスクドライブの残量に留意しください。

<https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz>

このデータをgunzipで展開すると、各行がスペースで区切られたCSV的なフォーマットのファイルが得られます。
内容は1行目だけはデータ行数と次元数、それ以降の行は1カラム目を単語とし、残りの300カラムはベクトルの各次元の値です。
これを前述のとおり、以下のようなフォーマットに整形します。

    {単語}{タブ文字}{JSON配列で表現された300次元のベクトル}

以下はその整形のための手続きです。

```console
# 1行目をスキップし、そのあとの2000行を取り出す
$ head -2001  cc.en.300.vec | tail +2 > 2e3.vec

# 単語部分を取り出す(1)
$ cut -f 1 -d ' ' 2e3.vec > 1.tmp

# ベクトル部分を取り出し、JSONの配列になるように整形(2)
$ cut -f 2- -d ' ' 2e3.vec | sed -e 's/ /,/g' -e 's/^/[/' -e 's/$/]/' > 2.tmp

# 1と2を連結して、目的のフォーマットにする
$ paste 1.tmp 2.tmp > 2e3.tsv

# 作業用の一時ファイルを消す
$ rm -f 1.tmp 2.tmp
```
