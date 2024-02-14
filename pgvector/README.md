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

# 
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

## メモ

* Max 2000 次元
    * 正確にはインデックス(IVFFlat, HNSW)が効くくのが 2000 次元まで
    * 格納だけなら 16000 次元まで
* 方式はHNSWもしくはIVF(転置ファイルインデックス)
* 各次元は `float` (32ビット)型で表現される

