# Item2Vec 関連、調べる

Collaborative Filtering (強調フィルタリング)という分野。
古典的なCFにはコールドスタート、多彩な特徴量の紙ができない、という問題がある。

アイテムをどうベクトル化するかという話しになる。

word2vecの応用でitem2vecできる。
文章の単語列をアイテムの閲覧履歴に置き換えて処理する。
word2vecの一部の構造はitem2vecでは不要になる。
ハイパーパラメータが異なるので注意。

論文 Item2Vec: Neural Item Embedding for Collaborative Filtering
https://qiita.com/FukuharaYohei/items/255d8df71bbf6b000ea4

論文メモ: Item2Vec: Neural Item Embedding for Collaborative Filtering
https://ohke.hateblo.jp/entry/2017/12/02/230000

item2vecだと、アイテム件数が多くなった時に指標が良くない。
https://engineering.mercari.com/blog/entry/20230612-cf-similar-item/

実装
* Python [gensim](https://pypi.org/project/gensim/) 自然言語処理
* Python [implicit](https://github.com/benfred/implicit) CF


ベクトル化をNNでやる場合、だいたい同じ構造・使い方になる。
学習データから変換ネットワーク≒モデルを得る。
モデルに対して実際のアイテムを通してベクトルを得る。

モデルの差し替えがカジュアルに行われている感。

GoogleのTwo towerが新し目。
https://www.youtube.com/watch?v=3giqIW2pIW4

> クエリと候補のペアからembedding抽出

Swivelモデル: 共起行列

Two-Towerの弱点: モデルの設計と学習が必要
