# sqlite-vss 入門

<https://github.com/asg017/sqlite-vss>

sqlite-vss とは、SQLiteでFaiss(ベクトル検索ライブラリ)を使えるようにする、SQLiteの拡張モジュールです。
SQLiteは言わずとも知れた組込み用のRDBMSで、多くの言語にバインディング・ドライバーが存在するとてもメジャーなRDBMSの1つです。
もう一方のFaissはFacebookが開発した、密なベクトルの効率的な類似性検索とクラスタリングのためのライブラリです。
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
INSERT INTO vss_word(rowid, vector) SELECT rowid, vector FROM word;
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
この例で指定しているのは`HNSW,Flat,IDMap`です。
この文字列はsqlite-vssからFaissの`faiss::index_factory()`に直接渡され、インデックスを作成するのに使われます。
Faissのindex factoryについてのドキュメントは下記のリンクを参照してください。

<https://github.com/facebookresearch/faiss/wiki/The-index-factory>

またより詳細な仕様を知るには`Faiss::index_factory()`のソースを見る必要があるかもしれません。
その場合は以下のリンクを参照してください。

<https://github.com/facebookresearch/faiss/blob/868e17f29493075742170885f1f57c7b9e61d9ea/faiss/index_factory.cpp#L883>
