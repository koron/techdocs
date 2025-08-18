# FreeBSD pkg and ports (and portmaster)

## ports & pkg environment

ユースケース毎のコマンド

ID    | ユースケース    | コマンド
:----:|-----------------|--------------------------------------------------------------
UC01  |インストール     | `portmaster -d {category}/{name}` <br> `pkg install {category}/{name}` <br> `make -C /usr/ports/{category}/{name} install`
UC02  |アンインストール | `portmaster -de {pkgname}` <br> `pkg delete {pkgname}`
UC03  |クリーンナップ   | `portmaster -ds` <br> `pkg autoremove`
UC04  |パッケージ更新   | `portmaster -da` <br> `pkg upgrade`
UC05  |レポジトリの同期 | `make -C /usr/ports update fetchindex` <br> `pkg update -f`
UC06  |未・一覧         | `awk -F\\| '{ print $2 }' /usr/ports/INDEX-14` <br> `pkg search '.*'` <br> `pkg rquery -a '%o'`
UC07  |未・詳細説明     | `pkg search {pkgname}` <br> `pkg rquery '%o' {pkgname}`
UC08  |未・依存一覧     | `awk -F\\| ' $1~/^{name}-/ { print $9 }' INDEX-14` <br> `pkg rquery %do {pkgname}`
UC09a |済・一覧         | `portmaster -l` 
UC09b |済・名前フィルタ | `pkg info`
UC09c |済・手動属性     | `pkg query -e '%a = 0' '%n-%v'`
UC09d |済・自動属性     | `pkg query -e '%a = 1' '%n-%v'`
UC10  |済・詳細説明     | `pkg info {pkgname}`
UC11  |済・構成一覧     | `pkg info -l {pkgname}`
UC12  |済・構成逆引き   | `pkg which {/path/to/file}`
UC13  |済・依存一覧     | `pkg info -d {pkgname}`
UC14  |済・被依存一覧   | `pkg info -r {pkgname}`
UC15a |済・手動属性付与 | `pkg set -A 0 {pkgname}`
UC15b |済・自動属性付与 | `pkg set -A 1 {pkgname}`

## pkg only environment

ユースケース毎のコマンド

ID    | ユースケース    | コマンド
:----:|-----------------|--------------------------------------------------------------
UC01  |インストール     | `pkg install {pkgname}`
UC02  |アンインストール | `pkg delete {pkgname}`
UC03  |クリーンナップ   | `pkg autoremove`
UC04  |パッケージ更新   | `pkg upgrade`
UC05  |レポジトリの同期 | `pkg update -f`
UC06  |未・一覧         | `pkg search '.*'` (インストール済も含む)
UC07  |未・詳細説明     | `pkg search -f {pkgname}`
UC08  |未・依存一覧     | `pkg rquery %do {pkgname}`
UC09a |済・一覧         | `pkg info`
UC09b |済・名前フィルタ | `pkg info \| grep {pattern}`
UC09c |済・手動属性     | `pkg query -e '%a = 0' '%n-%v'`
UC09d |済・自動属性     | `pkg query -e '%a = 1' '%n-%v'`
UC10  |済・詳細説明     | `pkg info {pkgname}`
UC11  |済・構成一覧     | `pkg info -l {pkgname}`
UC12  |済・構成逆引き   | `pkg which {/path/to/file}`
UC13  |済・依存一覧     | `pkg info -d {pkgname}`
UC14  |済・被依存一覧   | `pkg info -r {pkgname}`
UC15a |済・手動属性付与 | `pkg set -A 0 {pkgname}`
UC15b |済・自動属性付与 | `pkg set -A 1 {pkgname}`
