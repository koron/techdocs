# Merlin Transormers4Rec tutorial

https://github.com/NVIDIA-Merlin/Transformers4Rec

使うのはNGC CatalogのこのMerlin PyTorchイメージ。

https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-pytorch/tags

なんか突然ログイン(docker login)を求められた。
数週間前にpullしたイメージなのに。
Webで確認したらクラウドアカウントを作らされた。
その後Setup→Generate API KeysでNGC用のキーを作り…
`docker login nvcr.io`

`Username:` には文字通り `$oauthtoken` を
(特別なユーザー名らしい。 `git@github.com` みたいなものだろう。)
`Password:` にはキーを作った際のトークンを入力する。

あとは以下でコンテナを起動できる。

```
docker run --gpus all  --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_NICE nvcr.io/nvidia/merlin/merlin-pytorch:nightly /bin/bash
```

Jekyllを起動するのはログイン後に以下で

```
cd / ; jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token=''
```

チュートリアル: Getting Started Session Based をやってみた。
(パスはコンテナ内の /transformers4rec/examples/getting-started-session-based)

1. データ生成
2. 学習してモデル生成
3. [Triton Server](https://github.com/triton-inference-server/server) を起動して推論リクエストを投げる

という3段構成。

Triton Server は正式名称 Triton Inference Server で TIS と略されることがある。

以下、第三者が書いた関連記事

* <https://qiita.com/Getty708/items/b802a54f1f2e9926dfa6> (2023/12)
* <https://qiita.com/dcm_yamaya/items/985a57598d516e77894f> (2020/12)

TISは複数の計算資源を統括して複数のモデルを切り替えつつ利用できる推論サーバーと言った感じ。

疑問点

*   学習時にどこに注目して学習するかを指定する必要があるはず。どこでやってんの?
    *   データ作成時にタグ(ITEM, CATEGORY)を指定していた。それはスキーマに記録されている。
*   推論時にパラメーターを入れてるはず。どんな内容?
    *   半端なセッションデータ(最初に生成したもののミニ版)
*   推論結果が0フィルされてるように見えたが、なにか失敗してるのか? (未調査)

End to End Session Baseは推論時のリクエストを作るところでエラーになった。
詳細原因は不明だがデータ中にN/Aが混ざってて数値へ変換できない、と言われている。
