# Spotify Voyagerの調査

* [Spotify Voyager](https://spotify.github.io/voyager/)
* 同GitHub [spotify/voyager](https://github.com/spotify/voyager)

## Memo

インメモリ用の最近傍探索である。

実装はC++で、JavaとPython wrapperがある。

[maven repoで配布される](https://mvnrepository.com/artifact/com.spotify/voyager) jar には各OS向けのプリコンパイルバイナリが含まれている。
対応しているプラットフォームとアーキテクチャ: mac-aarch64, mac-x64, linux-aarch64, linux-x64, win-x64
linux-aarch64はちょっと珍しい印象だが、最近は当たり前なんだろうか?

[5分のデモ動画](https://www.youtube.com/watch?v=kOpL1NbvlM4):
Python
word2vec 10000 200d のデータ
インデックスの細かい設定は無しで使える。
保存したインデックスは言語を問わず利用できる。

ドキュメントは少なめ。
各言語のAPIドキュメントが主。

* [Javadoc](https://spotify.github.io/voyager/java/com/spotify/voyager/package-summary.html): ちょっとひどくない?w

近似方式はHNSWっぽい。
量子化方式についてはAPIからは不明。
C++のソース上はE4M3(指数4bit、仮数3ビット)のFP8っぽい。
Float32, Float8, E4M3 の3つのモードがあるっぽい。

[Spotify製のvoyagerをTitan Embeddings G1でやってみた](https://qiita.com/moritalous/items/86bd46d9690959a855d6)
日本人が試してみたログ。

追試してみるべきだろう。
