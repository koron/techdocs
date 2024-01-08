Rubyを自前で維持するよりdockerを使った方が楽

使うイメージは [asciidoctor/docker-asciidoctor](https://hub.docker.com/r/asciidoctor/docker-asciidoctor)

現時点の最新タグは `1.60`

こんなスクリプト(`docker_asciidoctor`)を用意するとdockerで実行しやすいだろう。

```sh
#!/bin/sh

set -eu

DOCKER_IMAGE="asciidoctor/docker-asciidoctor:1.60.0"

dir="$(pwd)"
case $(uname -s) in
  MSYS*) dir=$(cygpath -w $dir) ;;
esac

docker run --rm -it -v "${dir}:/documents/" "${DOCKER_IMAGE}" "$@"
```

実行する時はこんな感じになる。

```console
$ docker_asciidoctor asciidoctor-pdf -v \
    -v
    -D build/pdf \
    -a pdf-fontsdir=./resources/fonts \
    -a pdf-themesdir=./resources/themes \
    -a pdf-theme=a5book \
    book.adoc
```

Asciidoctor PDFはCJKの設定(テーマ)が無いので0から全部やる必要がある。

主なCJK用のテーマを作るドキュメントは [Create a CJK Theme](https://docs.asciidoctor.org/pdf-converter/latest/theme/cjk/) に書いてある。

テーマ作るポイント: フォントを用意する
    
フリーの日本語フォントの選択肢は少ないが、4種類くらいあると良い。

*   sans-serif系: [Noto Sans JP](https://fonts.google.com/noto/specimen/Noto+Sans+JP) (ゴシック、見出し向け)
*   serif系: [Noto Serif JP](https://fonts.google.com/noto/specimen/Noto+Serif+JP) (明朝、本文向け)
*   数式: [Noto Sans Math](https://fonts.google.com/noto/specimen/Noto+Sans+Math)
*   コード: 等幅フォント [UDEVGothicJP](https://github.com/yuru7/udev-gothic) 他

ブラウザと違ってBoldやItalicは個別に指定する必要あり。
ウェイトは normal (400)と bold (700)

サポートしているのはTTFとOTFのみ。
OTFもPostScript系のはNGなので、FontForgeでTTF系に変換する
(OTFにはTrueType系とPostScript系があるらしい)

変換例: `target_font.otf` から `target_font.ttf` が得られる。

```
$ fontforge -lang=ff -c 'Open($1); CIDFlatten(); Generate($1:r+".ttf"); Quit(0);' target_font.otf
```

Variable Fontは、ウェイトの設定が不明で thin になりがち。
