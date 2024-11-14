### apt (Ubuntu, Debian)

ユースケース毎のコマンド

ID    | ユースケース    | コマンド
:----:|-----------------|--------------------------------------------------------------
UC01  |インストール     | `apt install {pkgname}`
UC02  |アンインストール | `apt remove {pkgname}` <br> `apt purge {pkgname}`
UC03  |クリーンナップ   | `apt autoremove`
UC04  |パッケージ更新   | `apt upgrade`
UC05  |レポジトリの同期 | `apt update`
UC06  |未・一覧         | `apt list` <br> `LANG=C apt list \| grep -v installed`
UC07  |未・詳細説明     | `apt show {pkgname}`
UC08  |未・依存一覧     | `apt depends {pkgname}`
UC09a |済・一覧         | `apt list --installed`
UC09b |済・名前フィルタ | ???
UC09c |済・手動属性     | `apt-mark showauto`
UC09d |済・自動属性     | `apt-mark showmanual`
UC10  |済・詳細説明     | `apt show {pkgname}`
UC11  |済・構成一覧     | `dpkg -L {pkgname}`
UC12  |済・構成逆引き   | `dpkg -S {/path/to/file}`
UC13  |済・依存一覧     | `apt depends {pkgname}`
UC14  |済・被依存一覧   | `apt rdepends {pkgname}`
UC15a |済・手動属性付与 | `apt-mark manual {pkgname}`
UC15b |済・自動属性付与 | `apt-mark auto {pkgname}`
