# パッケージマネージャーへの要件

各OSにいろんなパッケージマネージャーがあるけど、
基本的な使い方を横断的に定義してメモしておこうという記事。
ベースになるのはFreeBSDのpkg。非常に整理されててわかりやすいので。

## 基礎的な要件

要件って言うかユースケースとかシナリオというほうが適切かも。

*   パッケージを指定してインストールできること

    必要な(依存する)別パッケージがあればそれらを自動でインストールする。
    指定したパッケージは非自動とマークし、
    依存により自動でインストールされたパッケージは自動とマークし、
    後にその情報を利用して、参照や削除ができること。

*   パッケージを指定してアンインストール(削除)できること

    依存パッケージについてはどちらでも良い。
    ただし自動で削除しないのであれば追加で以下の要件が生じる。

    *   どこからも依存されてないパッケージを検出し削除できること

*   未インストールのパッケージの情報を参照できること

    * 未インストールのパッケージの一覧を表示できる
    * 当該パッケージの詳細情報を参照できる
    * 当該パッケージの依存パッケージの一覧を参照できる。
    * (OPTION) 当該パッケージの依存パッケージの一覧を再帰的に参照できる
    * (OPTION) 当該パッケージの依存パッケージの一覧を再帰的に木構造で参照できる

*   インストール済みパッケージの情報を参照できること

    *   インストール済みパッケージの一覧を表示できる
    *   当該パッケージの詳細情報を参照できる
    *   当該パッケージの依存パッケージの一覧を参照できる。
    *   インストールされているファイルが所属するパッケージを特定する
    *   自動・非自動の属性変更
    *   (OPTION) 当該パッケージの依存パッケージの一覧を再帰的に参照できる
    *   (OPTION) 当該パッケージの依存パッケージの一覧を再帰的に木構造で参照できる

    自動・非自動のインストールマークに基づき、インストール済みパッケージに対して次の分類を決定できる。
    これは[FreeBSDにおける分類](https://docs.freebsd.org/en/books/handbook/ports/#portmaster)に基づいたもの。
    オプションでも良い。

    *   Root: 依存が無く、また被依存もないパッケージ
    *   Trunk: 依存は無いけども、被依存はあるパッケージ
    *   Branch: 依存が有り、被依存もあるパッケージ
    *   Leaf: 依存は有るが、被依存はないパッケージ

*   インストール済みパッケージを最新版へ更新できること

    *   パッケージ情報をリモートの最新版からローカルへ同期する
    *   更新可能な各パッケージを更新する

        この際、非自動・自動のマークが変更されてはいけない

### ユースケースを箇条書き

前述の要件は見通しが悪いので、ここに一覧としてまとめ直す。 
カッコ内はユースケースの管理のためのID。

*   インストール (UC01)
*   アンインストール (UC02)
    *   クリーンナップ (UC03)
*   インストール済みの更新 (UC04)
    *   レポジトリの同期 (UC05)
*   未インストールの検証
    *   一覧 (UC06)
        *   名前指定(グロブや正規表現)フィルタ
    *   詳細(説明等) (UC07)
    *   依存パッケージ一覧 (UC08)
*   インストール済みの検証
    *   一覧 (UC09)
        *   名前指定(グロブや正規表現)フィルタ
        *   自動・非自動インストール属性指定フィルタ
    *   詳細(説明等) (UC10)
    *   構成ファイル一覧 (UC11)
        *   ファイルからパッケージを特定 (UC12)
    *   依存パッケージ一覧 (UC13)
    *   被依存パッケージ一覧 (UC14)
    *   自動・非自動の属性変更 (UC15)

とりあえず全15個とする。

## 調べたい・まとめたいパッケージマネージャー

*   FreeBSD pkg & ports (portmaster)
*   pacman (MSYS2, Windows)
*   apt (Ubuntu, Debian)
*   dnf (Fedora)
*   (OPTION) winget

### FreeBSD pkg and ports (portmaster)

FreeBSDは他と比べると独特のパッケージシステムを持っている。
`pkg` はコンパイル済みバイナリを管理するパッケージマネージャー。
ports (正確にはコマンドではなく `make` を使ったシステム) はソースからのビルドを管理するツール。
portsは `make` で依存関係を含めてバイナリパッケージを作成して `pkg` でインストールする。
そのため切っても切り離せない。
また ports は `portmaster` 等のラッパーコマンドで便利に扱えるようになっている。

以下の表中の `{pkgname}` は `{category}/{name}` でも良いのだが、`{name}-{version}` 方式のパッケージ名やそのグロブ等の部分一致でも指定できる。場合によっては追加のオプションが必要になる。

レポジトリとして ports と pkg の2系統あるので、バージョン等が多少食い違うことがあることに注意が必要。

ユースケース | コマンド
:-----------:|--------------------------------------------------------------
UC01         | `portmaster -d {category}/{name}` <br> `pkg install {category}/{name}` <br> `make -C /usr/ports/{category}/{name} install`
UC02         | `portmaster -de {pkgname}` <br> `pkg delete {pkgname}`
UC03         | `portmaster -ds` <br> `pkg autoremove`
UC04         | `portmaster -da` <br> `pkg upgrade`
UC05         | `make -C /usr/ports update fetchindex` <br> `pkg update -f`
UC06         | `awk -F\\| '{ print $2 }' /usr/ports/INDEX-14` <br> `pkg search '.*'` <br> `pkg rquery -a '%o'`
UC07         | `pkg search {pkgname}` <br> `pkg rquery '%o' {pkgname}`
UC08         | `awk -F\\| ' $1~/^{name}-/ { print $9 }' INDEX-14` <br> `pkg rquery %do {pkgname}`
UC09         | `portmaster -l` <br> `pkg info` <br> `pkg query -e '%a = 0' '%n-%v` <br> `pkg query -e '%a = 1' '%n-%v'`
UC10         | `pkg info {pkgname}`
UC11         | `pkg info -l {pkgname}`
UC12         | `pkg which {/path/to/file}`
UC13         | `pkg info -d {pkgname}`
UC14         | `pkg info -r {pkgname}`
UC15         | `pkg set -A 0 {pkgname}` <br> `pkg set -A 1 {pkgname}`

### pacman (MSYS2, Windows)

ユースケース | コマンド
:-----------:|--------------------------------------------------------------
UC01         | `pacman -S {pkgname}`
UC02         | `pacman -R {pkgname}`
UC03         | `pacman -Qtdq | pacman -Rs -` <br> `pacman -Sc`
UC04         | `pacman -Su` <br> `pacman -Syu`
UC05         | `pacman -Sy`
UC06         | `pacman -Ss {regex}`
UC07         | `pacman -Si {pkgname}`
UC08         | `pacman -Si {pkgname}`
UC09         | `pacman -Q` <br> `pacman -Qs {regex}` <br> `pacman -Qe` <br> `pacman -Qd`
UC10         | `pacman -Qi {pkgname}`
UC11         | `pacman -Ql {pkgname}`
UC12         | `pacman -F {/path/to/file}`
UC13         | `pactree {pkgname}`
UC14         | `pactree -r [-d {n}] {pkgname}`
UC15         | `pacman -D --asexplicit {pkgname}` <br> `pacman -D --asdeps {pkgname}`

### apt (Ubuntu, Debian)

ユースケース | コマンド
:-----------:|--------------------------------------------------------------
UC01         | `apt install {pkgname}`
UC02         | `apt remove {pkgname}` <br> `apt purge {pkgname}`
UC03         | `apt autoremove`
UC04         | `apt upgrade`
UC05         | `apt update`
UC06         | `apt list` <br> `LANG=C apt list \| grep -v installed`
UC07         | `apt show {pkgname}`
UC08         | `apt depends {pkgname}`
UC09         | `apt list --installed` <br> `apt-mark showauto` <br> `apt-mark showmanual`
UC10         | `apt show {pkgname}`
UC11         | `dpkg -L {pkgname}`
UC12         | `dpkg -S {/path/to/file}`
UC13         | `apt depends {pkgname}`
UC14         | `apt rdepends {pkgname}`
UC15         | `apt-mark manual {pkgname}` <br> `apt-mark auto {pkgname}`

### dnf (Fedora)

ユースケース | コマンド
:-----------:|--------------------------------------------------------------
UC01         | `dnf install {pkgname}`
UC02         | `dnf remove {pkgname}`
UC03         | `dnf autoremove`
UC04         | `dnf upgrade`
UC05         | `dnf --refresh upgrade`
UC06         | `dnf list --available [pattern]`
UC07         | `dnf info [--available] {pkgname}`
UC08         | `dnf repoquery --requires {pkgname}`
UC09         | `dnf list --installed` <br> `dnf repoquery --userinstalled` <br> `dnf repoquery --installed --qf='%{name}-%{evr}.%{arch} (%{reason})\n' \| grep -v '(User)'`
UC10         | `dnf info [--installed] {pkgname}`
UC11         | `dnf repoquery [--installed] -l {pkgname}`
UC12         | `dnf repoquery [--installed] -f {/path/to/file}`
UC13         | `dnf repoquery --requires {pkgname}`
UC14         | `dnf repoquery --whatrequires {pkgname} --installed`
UC15         | `dnf mark install {pkgname}` <br> `dnf mark remove {pkgname}`

### (OPTION) winget

ユースケース | コマンド
:-----------:|--------------------------------------------------------------
UC01         |
UC02         |
UC03         |
UC04         |
UC05         |
UC06         |
UC07         |
UC08         |
UC09         |
UC10         |
UC11         |
UC12         |
UC13         |
UC14         |
UC15         |
