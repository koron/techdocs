# pgvector の実験

[pgvector/pgvector](https://github.com/pgvector/pgvector/) 触ってみた

## とりあえずの起動方法

```console
$ docker run --rm -it -p 5432:5432 -e LANG=C.UTF-8 -e POSTGRES_PASSWORD=abcd1234 pgvector/pgvector:0.6.0-pg16
```

`-pg16` は12から16が提供されている中から16を選んでいる。
以下で接続できる。
psqlのバージョンが低いのは許して。

```console
$ psql -U postgres
ユーザー postgres のパスワード:

psql (14.5、サーバー 16.1 (Debian 16.1-1.pgdg120+1))
警告: psql のメジャーバージョンは 14 ですが、サーバーのメジャーバージョンは 16 です。
         psql の機能の中で、動作しないものがあるかもしれません。
"help"でヘルプを表示します。

postgres=#
```

## もうちょっとまともな起動・接続・終了方法

```consonle
# 起動: コンテナに pgvector0 と命名してデーモンモードで起動する
$ docker run --name pgvector0 -d -p 5432:5432 -e LANG=C.UTF-8 -e POSTGRES_PASSWORD=abcd1234 pgvector/pgvector:0.6.0-pg16

# 停止: 止めただけではコンテナは残る。以下の再開もしくは削除が使える
$ docker stop pgvector0

# 再開
$ docker start pgvector0

# 削除: コンテナを削除する。データは消えるので要注意
$ docker rm pgvector0

# 接続: 稼働中のDBコンテナにexecして接続するのでパスワード不要
$ docker exec -it pgvector0 psql -U postgres
```

ローカルでpsqlが利用できる場合

```console
# 基本の接続はこれで良い
$ psql postgresql://postgres:abcd1234@localhost/postgres

# ちょっと使いやすくした設定で接続する方法
$ LANG=C.utf-8 psql -P "null=(null)" -P pager=0 postgresql://postgres:abcd1234@localhost/postgres
```

## Try: Getting Started

```sql
-- 拡張をインストール (1度だけ)
CREATE EXTENSION vector;

-- 3次元のベクトル
CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3));

-- サンプルデータ投入
INSERT INTO items (embedding) VALUES ('[1,2,3]'), ('[4,5,6]');

-- クエリ(L2)
SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;

-- クエリ(Inner Product(negative))
SELECT * FROM items ORDER BY embedding <#> '[3,1,2]' LIMIT 5;

-- クエリ(Cosine Distance)
SELECT * FROM items ORDER BY embedding <=> '[3,1,2]' LIMIT 5;
```

インデクシングをしていないのでフルスキャンになってるはず。

vector は `'[a, b, c, ... ]'` のようにJSONを文字列として渡すように見える。

```sql
UPDATE items SET embedding = '[1,2,3]' WHERE id = 1;
```

`<->`, `<#>`, `<=>` はそれぞれ距離を返すので、SELECTの選択カラムでも利用できる。

`AVG(embedding)` で平均が取れる。

## Indexing: HNSW

```sql
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops);
```

フォーマットはこんな感じ

```sql
CREATE INDEX {インデックス名} ON {テーブル名} USING hnsw ({対象カラム} {距離関数}) [WITH ({オプション})];

-- インデックス名: インデックスの識別名。条件次第で省略可能
-- テーブル名: インデックスを追加するテーブル名
-- 対象カラム: インデックスを追加するカラム名
-- 距離関数: vector_l2_ops, vector_ip_ops, vector_cosine_opts のいずれか
-- オプション:
--      m: レイヤー毎の最大接続数 (デフォルトは16)
--      ef_construction: グラフ構築時の動的候補リストのサイズ (デフォルトは64) 
```

`SET hnsw.ef_search = 100;`
検索時の動的候補リストサイズ
(デフォルト40)

`SET maintenance_work_mem = '8GB';`
インデックス作成時の利用可能メモリ
(デフォルト不明)

`SET max_parallel_maintenance_workers = 7;`
v0.6.0 以降は並列でインデックスを構築できる。
(デフォルト8)

`SELECT phase, round(100.0 * blocks_done / nullif(blocks_total, 0), 1) AS "%" FROM pg_stat_progress_create_index;`
PostgreSQL 12以降ではインデックス作成のプログレスを確認できる。
`phase` は `intitializing` と `loading tuples` の2つ。

## Indexing: IVFFlat

サンプル:

```sql
CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
```

```sql
CREATE INDEX {インデックス名} ON {テーブル名} USING ivfflat ({対象カラム} {距離関数}) [WITH ({オプション})];

-- 基本的にHNSWと同じ
-- オプション:
--      lists: 必須。リストの近似値。100万行までは `rows / 1000` を、それを越えたら `sqrt(rows)` が目安
```

`SET ivfflat.probes = 10;` 
検索時のプローブ数
(デフォルト1)

`SET max_parallel_maintenance_workers = 7;`
並列でインデックスを構築
(デフォルト8)

`SELECT phase, round(100.0 * tuples_done / nullif(tuples_total, 0), 1) AS "%" FROM pg_stat_progress_create_index;`
PostgreSQL 12以降ではインデックス作成のプログレスを確認できる。
`phase` は `intitializing`, `performing k-means`, `assigning tuples`, `loading tuples` の4つ。

## なんか便利そうなの

`SELECT pg_size_pretty(pg_relation_size('index_name'));`
インデックスのメモリサイズを確認。

## テスト

### まずは基本の実験

8次元10000個の単位ベクトル `vec_8d_10000_norm.tsv`

```console
$ go run ./gen_vec.go -norm -n 10000 > vec_8d_10000_norm.tsv
```

SQL読み込み

```sql
CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(8));

\COPY items(embedding) FROM 'vec_8d_10000_norm.tsv' WITH DELIMITER E'\t';  
```

問い合わせ

```sql
SELECT * FROM items WHERE id != 1 ORDER BY embedding <-> (SELECT embedding FROM items WHERE id = 1) LIMIT 10; 

  id  |                                   embedding
------+--------------------------------------------------------------------------------
  637 | [0.361608,0.423913,0.43722,-0.172323,-0.433289,-0.085588,0.150804,-0.500874]
 1712 | [0.402953,0.129779,0.506498,0.037384,-0.413361,-0.09904,0.319788,-0.529063]
 3818 | [0.610001,0.407145,0.140849,-0.041167,-0.417565,0.227718,0.142836,-0.440432]
 4874 | [0.462719,0.299788,0.480148,0.002447,-0.386452,0.297166,-0.126946,-0.460111]
 3303 | [0.439652,0.151159,0.405856,0.044084,-0.466334,-0.340497,0.236519,-0.477335]
 8489 | [0.402498,0.222113,0.614418,-0.029873,-0.159936,-0.10116,-0.070176,-0.607883]
 3449 | [0.600073,0.384041,0.246509,-0.072206,-0.550404,-0.172905,-0.049335,-0.301944]
 3815 | [0.577473,-0.008173,0.339957,0.181031,-0.338459,0.266005,-0.021752,-0.57648]
 7309 | [0.564374,-0.066568,0.470891,-0.069458,-0.16172,0.074873,0.302848,-0.571849]
 7156 | [0.441296,0.55304,0.164607,0.027719,-0.568404,0.158094,0.099399,-0.337022]
(10 rows)
```

インデックス抜きでの実行計画

```sql
EXPLAIN ANALYZE SELECT * FROM items WHERE id != 1 ORDER BY embedding <-> (SELECT embedding FROM items WHERE id = 1) LIMIT 10;

                                                            QUERY PLAN
-----------------------------------------------------------------------------------------------------------------------------------
 Limit  (cost=468.37..468.40 rows=10 width=53) (actual time=3.263..3.265 rows=10 loops=1)
   InitPlan 1 (returns $0)
     ->  Index Scan using items_pkey on items items_1  (cost=0.29..8.30 rows=1 width=37) (actual time=0.008..0.009 rows=1 loops=1)
           Index Cond: (id = 1)
   ->  Sort  (cost=460.07..485.07 rows=9999 width=53) (actual time=3.262..3.263 rows=10 loops=1)
         Sort Key: ((items.embedding <-> $0))
         Sort Method: top-N heapsort  Memory: 27kB
         ->  Seq Scan on items  (cost=0.00..244.00 rows=9999 width=53) (actual time=0.018..2.241 rows=9999 loops=1)
               Filter: (id <> 1)
               Rows Removed by Filter: 1
 Planning Time: 0.066 ms
 Execution Time: 3.281 ms
(12 rows)
```

HNSWインデックス作成、ちょっと時間かかる

```sql
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops);
```

再度問合せ

```sql
SELECT * FROM items WHERE id != 1 ORDER BY embedding <-> (SELECT embedding FROM items WHERE id = 1) LIMIT 10;

  id  |                                   embedding
------+--------------------------------------------------------------------------------
  637 | [0.361608,0.423913,0.43722,-0.172323,-0.433289,-0.085588,0.150804,-0.500874]
 1712 | [0.402953,0.129779,0.506498,0.037384,-0.413361,-0.09904,0.319788,-0.529063]
 3818 | [0.610001,0.407145,0.140849,-0.041167,-0.417565,0.227718,0.142836,-0.440432]
 4874 | [0.462719,0.299788,0.480148,0.002447,-0.386452,0.297166,-0.126946,-0.460111]
 3303 | [0.439652,0.151159,0.405856,0.044084,-0.466334,-0.340497,0.236519,-0.477335]
 8489 | [0.402498,0.222113,0.614418,-0.029873,-0.159936,-0.10116,-0.070176,-0.607883]
 3449 | [0.600073,0.384041,0.246509,-0.072206,-0.550404,-0.172905,-0.049335,-0.301944]
 3815 | [0.577473,-0.008173,0.339957,0.181031,-0.338459,0.266005,-0.021752,-0.57648]
 7309 | [0.564374,-0.066568,0.470891,-0.069458,-0.16172,0.074873,0.302848,-0.571849]
 7156 | [0.441296,0.55304,0.164607,0.027719,-0.568404,0.158094,0.099399,-0.337022]
(10 rows)
```

インデックス有りでの実行計画。
シンプルになってる。
最大468.40だったコストが25.43で1/18。
Execution Timeは3.281msが0.334ms1/10。

```sql
EXPLAIN ANALYZE SELECT * FROM items WHERE id != 1 ORDER BY embedding <-> (SELECT embedding FROM items WHERE id = 1) LIMIT 10;

                                                               QUERY PLAN

-----------------------------------------------------------------------------------------------------------------------------------------
 Limit  (cost=24.90..25.43 rows=10 width=53) (actual time=0.307..0.316 rows=10 loops=1)
   InitPlan 1 (returns $0)
     ->  Index Scan using items_pkey on items items_1  (cost=0.29..8.30 rows=1 width=37) (actual time=0.011..0.011 rows=1 loops=1)
           Index Cond: (id = 1)
   ->  Index Scan using items_embedding_idx on items  (cost=16.60..542.60 rows=9999 width=53) (actual time=0.306..0.313 rows=10 loops=1)
         Order By: (embedding <-> $0)
         Filter: (id <> 1)
         Rows Removed by Filter: 1
 Planning Time: 0.092 ms
 Execution Time: 0.334 ms
(10 rows)
```

Inner Product と Cosine でも同様にHNSWインデックスの有無で、実行計画の同様の変化を確認した。

テーブルとインデックスのサイズを確認。
素のベクトルだけで 4byte * 8dim * 10000vecs = 320kB 必要。
テーブルサイズはおおよそそのオーダーなのに対し、インデックスは1つあたりその4倍くらい。

```
postgres=# SELECT pg_size_pretty(pg_relation_size('items'));
 pg_size_pretty
----------------
 752 kB
(1 row)

postgres=# SELECT pg_size_pretty(pg_relation_size('items_embedding_idx'));
 pg_size_pretty
----------------
 3272 kB
(1 row)

postgres=# SELECT pg_size_pretty(pg_relation_size('items_embedding_idx_cos'));
 pg_size_pretty
----------------
 3280 kB
(1 row)

postgres=# SELECT pg_size_pretty(pg_relation_size('items_embedding_idx_ip'));
 pg_size_pretty
----------------
 3280 kB
(1 row)
```

### ベクトル数を多くしてみる

2倍の20000ベクトルのデータを作って同じことをしてみる。

```
DROP TABLE items;

$ go run ./gen_vec.go -n 20000 > vec_8d_20000.tsv

CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(8));

\COPY items(embedding) FROM 'vec_8d_20000.tsv' WITH DELIMITER E'\t';  

SELECT * FROM items WHERE id != 1 ORDER BY embedding <-> (SELECT embedding FROM items WHERE id = 1) LIMIT 10; 
```

インデックスの無い状態で実行計画を取ると、10000件の時のおおよそ倍の時間がかかる。
これは期待した通り。

```
EXPLAIN ANALYZE SELECT * FROM items WHERE id != 1 ORDER BY embedding <-> (SELECT embedding FROM items WHERE id = 1) LIMIT 10;

                                                            QUERY PLAN
-----------------------------------------------------------------------------------------------------------------------------------
 Limit  (cost=932.47..932.50 rows=10 width=53) (actual time=6.605..6.608 rows=10 loops=1)
   InitPlan 1 (returns $0)
     ->  Index Scan using items_pkey on items items_1  (cost=0.29..8.30 rows=1 width=37) (actual time=0.010..0.011 rows=1 loops=1)
           Index Cond: (id = 1)
   ->  Sort  (cost=924.17..974.17 rows=19999 width=53) (actual time=6.604..6.605 rows=10 loops=1)
         Sort Key: ((items.embedding <-> $0))
         Sort Method: top-N heapsort  Memory: 27kB
         ->  Seq Scan on items  (cost=0.00..492.00 rows=19999 width=53) (actual time=0.047..4.441 rows=19999 loops=1)
               Filter: (id <> 1)
               Rows Removed by Filter: 1
 Planning Time: 0.066 ms
 Execution Time: 6.623 ms
(12 rows)
```

インデックスを作る。
作るのにかかった時間は少し伸びた感じ。

```
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops);
```

インデックス作成後のコスト構造は件数の影響がほぼ見えない。
Execution Timeは `+log(2)` くらいの影響だろうか?
データ量2倍に対して 1.35 倍くらいの処理時間。

    0.453 / 0.334 = 1.356


```
EXPLAIN ANALYZE SELECT * FROM items WHERE id != 1 ORDER BY embedding <-> (SELECT embedding FROM items WHERE id = 1) LIMIT 10;

                                                                QUERY PLAN

-------------------------------------------------------------------------------------------------------------------------------------------
 Limit  (cost=24.91..25.44 rows=10 width=53) (actual time=0.430..0.439 rows=10 loops=1)
   InitPlan 1 (returns $0)
     ->  Index Scan using items_pkey on items items_1  (cost=0.29..8.30 rows=1 width=37) (actual time=0.005..0.006 rows=1 loops=1)
           Index Cond: (id = 1)
   ->  Index Scan using items_embedding_idx on items  (cost=16.60..1084.60 rows=19999 width=53) (actual time=0.430..0.438 rows=10 loops=1)
         Order By: (embedding <-> $0)
         Filter: (id <> 1)
         Rows Removed by Filter: 1
 Planning Time: 0.127 ms
 Execution Time: 0.453 ms
(10 rows)
```

テーブルサイズとインデックスサイズはほぼ倍

```
postgres=# SELECT pg_size_pretty(pg_relation_size('items'));
 pg_size_pretty
----------------
 1536 kB
(1 row)

postgres=# SELECT pg_size_pretty(pg_relation_size('items_embedding_idx'));
 pg_size_pretty
----------------
 6536 kB
(1 row)
```

### 次元数を多くしてベクトル数を減らす

1000次元 1000ベクトル 非正規化

```
DROP TABLE items;

CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(1000));

\COPY items(embedding) FROM 'vec_1000d_1000.tsv' WITH DELIMITER E'\t';

CREATE INDEX ON items USING hnsw (embedding vector_l2_ops);

postgres=# SELECT pg_size_pretty(pg_relation_size('items'));
 pg_size_pretty
----------------
 64 kB
(1 row)

postgres=# SELECT pg_size_pretty(pg_relation_size('items_embedding_idx'));
 pg_size_pretty
----------------
 8008 kB
(1 row)
```

テーブルサイズは4MBくらいのはずなんだけど、なんか小さい。
なんか圧縮かかってるんだろうか?
一方でHNSWインデックスは件数のわりに大きく感じる。
が、まぁこれはこちらが正常である感じも強い。

## `pg_relation_size` の検証

`pg_relation_size` 及び `pg_total_relation_size` でテーブルサイズを再度検証する。

```SQL
SELECT pg_size_pretty(pg_relation_size('items')) AS size, pg_size_pretty(pg_total_relation_size('items')) AS total;
```

まずは8次元1万データでの結果:

```
postgres=# SELECT pg_size_pretty(pg_relation_size('items')) AS size, pg_size_pretty(pg_total_relation_size('items')) AS total;
  size  |  total
--------+---------
 752 kB | 1032 kB
(1 row)
```

次に1000次元1000データでの結果:

```
postgres=# CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(1000));
CREATE TABLE
postgres=# \COPY items(embedding) FROM 'tmp/vec_1000d_1000.tsv' WITH DELIMITER E'\t';
COPY 1000
postgres=# SELECT pg_size_pretty(pg_relation_size('items')) AS size, pg_size_pretty(pg_total_relation_size('items')) AS total;
 size  |  total
-------+---------
 64 kB | 5584 kB
(1 row)
```

sizeとtotalの乖離が大きい。
生データは 4 * 1000 * 1000 ≒ 4MB なので total の 5.5MB に極めて近い。
`pg_relation_size` が出しているものが物理的なテーブルのサイズではないと考えるほうが妥当そう。
その裏付けを取る。

以下に[表9.94 データベースオブジェクトサイズ関数](https://www.postgresql.jp/document/15/html/functions-admin.html#FUNCTIONS-ADMIN-DBSIZE)より引用する。

> `pg_relation_size ( relation regclass [, fork text ] ) → bigint`
> 
> 指定したリレーションの一つの「fork」で使用されているディスクスペースを計算します。 （大抵の目的には、すべてのフォークのサイズを合計する高レベルの `pg_total_relation_size` あるいは `pg_table_size` を使う方が便利です。）

> `pg_total_relation_size ( regclass ) → bigint`
>
> 指定テーブルが使用している、インデックスとTOASTデータを含む全ディスクスペースを計算します。 結果は `pg_table_size + pg_indexes_size` と等価です。 

> `pg_table_size ( regclass ) → bigint`
>
> 指定テーブルが使用している、インデックスを含まないディスクスペースを計算します。（ただしあればTOASTテーブル、空き領域マップ、可視性マップを含みます。）

> `pg_indexes_size ( regclass ) → bigint`
>
> 指定したテーブルに付与されたインデックスで使用されている全ディスクスペースを計算します。

計測方法を再検証して確定し、ここまでの結果も計測しなおす。
まずは 1000d x 1000 にインデックスを与える。
その後 `pg_relation_size`, `pg_total_relation_size`, `pg_table_size`, `pg_indexes_size` を `items` 及び `items_embedding_idx` に適用して比較する。
結果をまとめたのが以下の表。

Function                 |`items` | `items_embedding_idx`
-------------------------|-------:|---------------------:
`pg_relation_size`       |64 kB   |8008 kB
`pg_total_relation_size` |13 MB   |8008 kB
`pg_table_size`          |5544 kB |8008 kB
`pg_indexes_size`        |8048 kB |0 kB

テーブルの状態は以下の通り。
テーブル `items` には `items_embedding_idx` 以外にも `items_pkey` があり `pg_indexes_size` の値にはそれが含まれていそう。

```
postgres=# \d items
                                Table "public.items"
  Column   |     Type     | Collation | Nullable |              Default
-----------+--------------+-----------+----------+-----------------------------------
 id        | bigint       |           | not null | nextval('items_id_seq'::regclass)
 embedding | vector(1000) |           |          |
Indexes:
    "items_pkey" PRIMARY KEY, btree (id)
    "items_embedding_idx" hnsw (embedding vector_l2_ops)
```

以下の指針で情報を取得することにする。

* 純粋なテーブルだけのサイズであれば `pg_table_size`
* 個別のインデックスであれば `pg_relation_size`
* 以下余談
    * テーブルに紐づいた全インデックスであれば `pg_indexes_size`
    * インデックスを含めたテーブル全体のサイズであれば `pg_total_relation_size`

以上からテーブルサイズおよびインデックスサイズ取得用のSQLは次の通りとする。

```sql
-- テーブル用
SELECT pg_size_pretty(pg_table_size('items'));

-- 個別インデックス用
SELECT pg_size_pretty(pg_relation_size('items_embedding_idx'));
```

当然のことだがインデックスをドロップしてもテーブルサイズは変わらない。
以下はそのことを示すログ。
そもそもインデックスがなくなったのでサイズ情報にアクセスできずエラーになった。
ログではテーブルだけに絞って再計測している。

```
postgres=# SELECT pg_size_pretty(pg_table_size('items')) AS table , pg_size_pretty(pg_relation_size('items_embedding_idx')) AS index;
  table  |  index
---------+---------
 5544 kB | 8008 kB
(1 row)

postgres=# DROP INDEX items_embedding_idx;
DROP INDEX
postgres=# SELECT pg_size_pretty(pg_table_size('items')) AS table , pg_size_pretty(pg_relation_size('items_embedding_idx')) AS index;
ERROR:  relation "items_embedding_idx" does not exist
LINE 1: ...ems')) AS table , pg_size_pretty(pg_relation_size('items_emb...
                                                             ^
postgres=# SELECT pg_size_pretty(pg_table_size('items')) AS table;
  table
---------
 5544 kB
(1 row)
```

### 再計測結果

次元数と行数によるテーブルサイズおよびインデックスサイズは以下の通り:

Dim     |Row    |Table Size |HNSW Size  |IVF Size
-------:|------:|----------:|----------:|----------:
   8    |10000  | 792 kB    |3280 kB    | 576 kB
   8    |20000  |1576 kB    |6536 kB    |1120 kB
1000    | 1000  |5544 kB    |8008 kB    |4016 kB

<details>
<summary>計測時のログ</summary>

```
postgres=# CREATE TABLE items8d10k (id bigserial PRIMARY KEY, embedding vector(8));
CREATE TABLE
postgres=# CREATE TABLE items8d20k (id bigserial PRIMARY KEY, embedding vector(8));
CREATE TABLE
postgres=# CREATE TABLE items1kd1k (id bigserial PRIMARY KEY, embedding vector(1000));
CREATE TABLE
postgres=# \d
                List of relations
 Schema |       Name        |   Type   |  Owner
--------+-------------------+----------+----------
 public | items1kd1k        | table    | postgres
 public | items1kd1k_id_seq | sequence | postgres
 public | items8d10k        | table    | postgres
 public | items8d10k_id_seq | sequence | postgres
 public | items8d20k        | table    | postgres
 public | items8d20k_id_seq | sequence | postgres
(6 rows)

postgres=# \COPY items8d10k(embedding) FROM 'vec_8d_10000_norm.tsv' WITH DELIMITER E'\t';
COPY 10000
postgres=# \COPY items8d20k(embedding) FROM 'vec_8d_20000.tsv' WITH DELIMITER E'\t';
COPY 20000
postgres=# \COPY items1kd1k(embedding) FROM 'tmp/vec_1000d_1000.tsv' WITH DELIMITER E'\t';
COPY 1000
postgres=# SELECT pg_size_pretty(pg_table_size('items8d10k'));
 pg_size_pretty
----------------
 792 kB
(1 row)

postgres=# SELECT pg_size_pretty(pg_table_size('items8d20k'));
 pg_size_pretty
----------------
 1576 kB
(1 row)

postgres=# SELECT pg_size_pretty(pg_table_size('items1kd1k'));
 pg_size_pretty
----------------
 5544 kB
(1 row)

postgres=# CREATE INDEX ON items8d10k USING hnsw (embedding vector_l2_ops);
CREATE INDEX
postgres=# CREATE INDEX ON items8d20k USING hnsw (embedding vector_l2_ops);
CREATE INDEX
postgres=# CREATE INDEX ON items1kd1k USING hnsw (embedding vector_l2_ops);
CREATE INDEX
postgres=# SELECT pg_size_pretty(pg_relation_size('items8d10k_embedding_idx'));
 pg_size_pretty
----------------
 3280 kB
(1 row)

postgres=# SELECT pg_size_pretty(pg_relation_size('items8d20k_embedding_idx'));
 pg_size_pretty
----------------
 6536 kB
(1 row)

postgres=# SELECT pg_size_pretty(pg_relation_size('items1kd1k_embedding_idx'));
 pg_size_pretty
----------------
 8008 kB
(1 row)

postgres=# CREATE INDEX ON items8d10k USING ivfflat (embedding vector_l2_ops) WITH (lists = 10);
CREATE INDEX
postgres=# \d items8d10k
                               Table "public.items8d10k"
  Column   |   Type    | Collation | Nullable |                Default
-----------+-----------+-----------+----------+----------------------------------------
 id        | bigint    |           | not null | nextval('items8d10k_id_seq'::regclass)
 embedding | vector(8) |           |          |
Indexes:
    "items8d10k_pkey" PRIMARY KEY, btree (id)
    "items8d10k_embedding_idx" hnsw (embedding vector_l2_ops)
    "items8d10k_embedding_idx1" ivfflat (embedding) WITH (lists='10')

postgres=# CREATE INDEX ON items8d20k USING ivfflat (embedding vector_l2_ops) WITH (lists = 20);
CREATE INDEX
postgres=# CREATE INDEX ON items1kd1k USING ivfflat (embedding vector_l2_ops) WITH (lists = 1);
CREATE INDEX
postgres=# SELECT pg_size_pretty(pg_relation_size('items8d10k_embedding_idx1'));
 pg_size_pretty
----------------
 576 kB
(1 row)

postgres=# SELECT pg_size_pretty(pg_relation_size('items8d20k_embedding_idx1'));
 pg_size_pretty
----------------
 1120 kB
(1 row)

postgres=# SELECT pg_size_pretty(pg_relation_size('items1kd1k_embedding_idx1'));
 pg_size_pretty
----------------
 4016 kB
(1 row)

```
</details>

## HNSWの比較

HNSWの実装を比較してみる。
といっても論文を元にしたアルゴリズムなので本質は変わらない。

pgvector の HNSW 実装には以下のパラメーターがあった。
他の実装にも同様のパラメーターがあったので横断的に再解釈してみる。
また差異もあったはずなのでそちらも比較対象とする。

```
--      m: レイヤー毎の最大接続数 (デフォルトは16)
--      ef_construction: グラフ構築時の動的候補リストのサイズ (デフォルトは64) 
```

[JavaのHNSW実装](https://github.com/jelmerk/hnswlib) には同様のパラメーターがなかった。

[原論文](https://arxiv.org/abs/1603.09320) からの抜粋

> `INSERT(hnsw, q, M, Mmax, efConstruction, mL)`
>
> Input: multilayer graph hnsw, new element q, number of established connections M, maximum number of connections for each element per layer Mmax, size of the dynamic candidate list efConstruction, normalization factor for level generation mL


[nsmlibのHNSW実装](https://github.com/nmslib/nmslib/blob/ade4bcdc9dd3719990de2503871450b8a62df4a5/similarity_search/src/method/hnsw.cc#L198-L208)

HNSWでは 1つのノードに対して各レイヤーで最大 `M` の隣接ノードへのリンクを持つ。
この隣接ノードを決定する際に少し多めに `ef_construction` 個のノードを探索する。
探索はエントリーポイントとなるノードを1つ与え、そこから隣接ノードを幅優先で辿り `ef_construction` 個のノードを辿ったところで打ち切られる。
`ef_construction` 個の候補ノードから `M` 個の隣接ノードを選ぶ方法はいくつかあるようだ。
その方法の例としては以下の通り:

1. 単により近いものから最大 `M` 個を選ぶ
2. より近いもの、かつ確定した隣接ノードに近すぎないものから最大 `M` 個を選ぶ
3. 2で `M` 個に満たない場合、確定した隣接ノードに近すぎたものからより近い順に `M` 個になるように選ぶ
4. 3に加えてリストを優先・通常・一時の合計3つに分けるもの。一時リストに近いものは通常リストに、優先や通常リストに近いものは一時リストに、それ以外は優先リストに分類する。

以上を踏まえて pgvector の実装を確認する。

* 論文通りナイーブに実装している雰囲気がある
* `M` は layer#0 においては 2倍になる
* 隣接ノードの選び方は戦略 3っぽい。4も実装されているがグラフビルド時に使われなさそう

念のためFaissの実装も見ておく。

* <https://github.com/facebookresearch/faiss/blob/main/faiss/impl/HNSW.h>
* <https://github.com/facebookresearch/faiss/blob/main/faiss/impl/HNSW.cpp>

やってることはほぼほぼ nmslib の実装と一緒。おそらく論文通りなのだろう。
採用している隣接ノード選択アルゴリズムは 3 で、他のものは実装していないようだ。

## メモ

* Max 2000 次元
    * 正確にはインデックス(IVFFlat, HNSW)が効くくのが 2000 次元まで
    * 格納だけなら 16000 次元まで
* 方式はHNSWもしくはIVF(転置ファイルインデックス)
* 各次元は `float` (32ビット)型で表現される
