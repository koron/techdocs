# goplsがGOROOTセンシティブである県の調査

.vim-lsp-settings/settings.json を作成し、以下のようにgoplsにログを吐かせる。

```json
{
    "gopls": {
        "args": [
            "-logfile=D:\\tmp\\gopls.log",
            "-vv"
        ]
    }
}
```

ログファイルは作られたが中身が出力されない。
`-logfile` が制御するのはwindow/logMessage の出力先だけだった。

リモートアプローチに変更。 `-listen=127.0.0.1:49999` で改変版サーバーを起動し、 `-remote=127.0.0.1:49999` で接続する。
改変版では普通にlogパッケージ使ってprintfデバッグをする。

GOROOTが小文字統一≒実際のパスと大文字小文字が異なる時、
GOROOT下のファイルに対して snapshot.MetadatForFileが0を返すことを確認した。
サーバーには正しいGOROOTが設定されていることから、
クライアントが送ってきたGOROOTを解釈に使っていると考えられる。

怪しいレスポンスを発見した。
`textDocument/definition` に対してレスポンスのURIが小文字のGOROOTベースの値を返している。

```json
{
  "request": {
    "id": 3,
    "jsonrpc": "2.0",
    "method": "textDocument/definition",
    "params": {
      "position": {
        "character": 35,
        "line": 9
      },
      "textDocument": {
        "uri": "file:///D:/home/koron/work/techdocs/how-GOROOT-determined/test1.go"
      }
    }
  },
  "response": {
    "id": 3,
    "jsonrpc": "2.0",
    "result": [
      {
        "range": {
          "end": {
            "character": 11,
            "line": 100
          },
          "start": {
            "character": 5,
            "line": 100
          }
        },
        "uri": "file:///D:/go/current/src/os/env.go"
      }
    ]
  }
}
```

念のため正常な動作も見てみるか。
正しいURI(`#/response/result/uri`)が返ってきてた。

```json
{
  "request": {
    "id": 3,
    "jsonrpc": "2.0",
    "method": "textDocument/definition",
    "params": {
      "position": {
        "character": 35,
        "line": 9
      },
      "textDocument": {
        "uri": "file:///D:/home/koron/work/techdocs/how-GOROOT-determined/test1.go"
      }
    }
  },
  "response": {
    "id": 3,
    "jsonrpc": "2.0",
    "result": [
      {
        "range": {
          "end": {
            "character": 11,
            "line": 100
          },
          "start": {
            "character": 5,
            "line": 100
          }
        },
        "uri": "file:///D:/Go/current/src/os/env.go"
      }
    ]
  }
}
```

`snapshot.meta (metadataGraph)` に小文字URIで格納されていることを確認。
textDocument/definitionで os/env.go を解決する際に、
誤ったGOROOTベースのURIで登録することが原因であると判明。

mapPositionが間違ったURIで呼ばれているが、呼び出し元とタイミングが不明。
関数の呼び出し元を返す簡易な仕組みが欲しい。

x/tools/go/packages.Load() に渡すConfig.Env に GOROOT が設定されている。
それに引きずられ packages.Load() が返すPackage.GoFilesのURIが影響を受ける。
このGOROOTはクライアントの環境変数由来である可能性が高い。

go/packagesでは `go list` を使ってパッケージの情報を集めてる。
その際に Config.Env をコマンドに引き継いでいる。
結果 `go list` の出力の Package.Dir がGOROOTに依存して変換する。
コマンドラインで模式的に状況を示すと、以下のようになる。

```console
$ GOROOT='D:\Go\current' go list -f '{{.Dir}}' os
D:\Go\current\src\os

$ GOROOT='D:\go\current' go list -f '{{.Dir}}' os
D:\go\current\src\os
```

このgo/packagesはPackage.Dirを使ってフルパスでGoFilesを保存する。

したがってGOROOTを小文字統一にしていると、IDであるURIが小文字になってしまうが
クライアント(Vim & vim-lsp)は大文字で問い合わせるため
ファイル及びファイルが見つけられなくなってしまう。

gopls localからgopls remoteへどうやって環境変数を伝えてるかも問題。

`initialize` メソッドの initializationOptions.env にGOXXXな環境変数を乗せてた。
サーバー側は受け取ってた。
クライアント側はaddGoEnvToInitializeRequestでFowarder.handlerで埋め込んでた。

Vimは GetLongPathName を使って正規化している可能性がある。
Goでもできた。test4.go参照
