# systemd の基礎に再入門

目的: RaspberryPi Zero 2 WにインストールしたRaspberry Pi OSの起動速度を上げるために不要そうなユニットを無効化する。

参考資料: [Raspberry Pi Zero W で電源 ON 後，最速で撮影画像をアップロード](https://rabbit-note.com/2019/05/05/raspberry-pi-zero-w-fast-boot/)

## コマンド

### `list-units` ユニット一覧を表示する

使い方:

```console
$ systemctl list-units
```

ユニット一覧を表示する。表示項目は UNIT, LOAD, ACTIVE, SUB, DESCRIPTION で意味は下記の通り。

* UNIT: ユニット名
* LOAD: 読み込み状態 (可能な値: loaded, not-found, bad-setting, error, masked)
* ACTIVE: 現在の動作状態 (可能な値: active, reloading, inactive, failed, activating, deactivating)
* SUB: 動作状態の副状態 (可能な値: ユニットタイプごとに違い `systemctl --state=help` で確認できる)
* DESCRIPTION: 説明

`--all` で全部のユニットを表示する。デフォルトはアクティブ、保留中のジョブがある、失敗と記録されたユニット。

引数でパターン(グロブ)を与えるとマッチするユニットだけを表示する。
また `--type={LOAD value}` や `--state={ACTIVE or SUB value}` で別途フィルタできる。

ヘッダーと「最後のユニット(何を指すか不明、依存関係ツリーの末端という意味だろうか?)」にはアンダーラインがひかれる。
先頭 UNIT の前の色付きの `●` はなんらかの理由で実行できなかった/しなかったユニットを示す。

systemctl をサブコマンド指定なしで実行するとデフォルトでこれが実行される。

### `status [PATTERN...|JOB...]` ユニットの詳細な情報を表示する

システム全体もしくは指定したユニットの、ランタイム状態情報を表示する。
ジャーナルに記録された直近のログも併せて表示する。

オプション:

* `--type=`
* `--state=`
* `--failed`
* `--all` 全体のツリーの後に各ユニットの詳細が表示される

人が読むために設計されている。機械的に読むなら show を使うこと。

アクセスできるのは現在の状態。
過去の状態や履歴が知りたいなら `journalctl` を使う。

色付きの `●` は状態が一目でわかるように。
緑の●はactive, 白の○はinactiveかmaintenance, 白の●はdeactivating,
赤の×はfailedかerror, 緑の↻はreloading

`Loaded` はメモリに読み込み済みかを示す。含まれうる値は loaded, error, not-found, bad-setting, masked.
(前述の list-units の `LOAD` と一緒だね)

`Active` は list-units の `ACTIVE` と一緒

### `list-dependencies` ユニットの依存関係を表示する

ユニットを指定しなければ全体の、指定されたらそのユニットに依存する、依存関係が表示される。
(ちょっと依存関係逆かも?)

依存関係を規定するのは定義ファイルの `Requires=`, `Requisite=`, `ConsistsOf=`, `Wants=`, `BindsTo=` あたり。
今回は定義ファイルの内容までは踏み込まないつもりだが、例外的に記載した。

* `--reverse` 依存関係を逆にたどって表示する
* `--after` 指定したユニット以降の依存関係 (デフォルト)
* `--before` 指定したユニット以前の依存関係

表示できるのはメモリにロードされた分だけ。
全体の依存関係ではないことに留意が要る。

### 実行・停止・再読み込み

* `start` ユニットを開始する
* `stop` ユニットを停止する
* `reload` 設定ファイルをリロードする。ユニット定義ファイルではないことに注意が必要
* `restart` ユニットを停止・開始する
* `try-restart` restartを一緒だが、ユニットがもともと停止していた場合はなにもしない
* `reload-or-restart` ユニットがreloadをサポートしてればreloadする。してない場合は restart する
* `try-reload-or-restart` reload-or-restartと同じだが、ユニットがもともと停止していた場合は何もしない
* `kill` ユニットのプロセスにシグナルを送る。 `--signal=` で送るシグナルを選べる

### ユニット定義に対するコマンド

* `list-unit-files` ユニットファイルの一覧と対応するユニットの現在の状態を表示する
* `enable` ユニットを有効とマークする
* `disable` ユニットを無効とマークする

### その他のコマンド

* `list-automounts` ユニットにより提供されるautomountの一覧を表示する
* `list-sockets` ユニットにより提供されるsocketの一覧を表示する
* `list-timers` ユニットにより提供されるタイマーの一覧を表示する
* `is-active` ユニットがアクティブ状態か調べる
* `is-failed` ユニットが失敗状態か調べる
* `show`
* `cat` ユニットの定義ファイルを表示する。いちいちstatusのLoadedのパスを見なくても良い。
* `help` ユニットのマニュアルを表示する。マニュアルが提供されない場合もある。
* `isolate` 指定したユニットとその依存ユニットを開始し、それ以外を止める。危険
* `clear` ユニットが使ってる persist なデータを消す?
* `freeze` 同一cgroupのユニットをsuspendする。プロセスが止まるらしい
* `thaw` freeze の逆操作
* `set-property` ユニットのプロパティを設定する
* `bind` ホストのパスをユニットのマウントスペースにバインドする
* `mount-image` ホストのイメージをユニットのマウントスペースにマウントする
* `service-log-level`
* `service-log-target`
* `reset-failed` ユニットの failed 状態をリセットする

## まとめ: 不要そうなユニットを無効化する、典型的なユースケース

無効化する場合: `sudo systemctl disable {ユニット名}`
以後の起動で無効化したユニットは実行されない。

ユニットのプロセスやタイマーなどを即時に止めたければ: `sudo systemctl stop {ユニット名}`

そもそもの話、どんなユニットが実行中(もしくは完了済み)なのかを知るには:
`systemctl list-units` もしくは単に `systemctl`

ユニットが何をするものなのか定義を知りたい場合: `systemctl cat {ユニット名}`

上記に `Documentation` があれば `systemctl help {ユニット名}` で詳しいドキュメントが読める。

ユニットの依存関係を見たい場合: `systemctl list-dependencies`
`--reverse` `--after` `--before` などのオプションを使って検証できる。

それぞれのユニットがenable/disableどっちなのかを知るには:
`systemctl list-unit-files`
デフォルト設定と現在の設定を比較できる。

## その他資料

* [RaspberryPi Zero 2 W でいろいろ試した際のメモ](./examine-rpz2.md)
