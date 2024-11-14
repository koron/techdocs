# pacman (MSYS2, Windows)

ユースケース毎のコマンド

ID    | ユースケース    | コマンド
:----:|-----------------|-----------------------------------------------
UC01  |インストール     | `pacman -S {pkgname}`
UC02  |アンインストール | `pacman -R {pkgname}`
UC03  |クリーンナップ   | `pacman -Qtdq | pacman -Rs -` <br> `pacman -Sc`
UC04  |パッケージ更新   | `pacman -Su` <br> `pacman -Syu`
UC05  |レポジトリの同期 | `pacman -Sy`
UC06  |未・一覧         | `pacman -Ss {regex}`
UC07  |未・詳細説明     | `pacman -Si {pkgname}`
UC08  |未・依存一覧     | `pacman -Si {pkgname}` (詳細に含まれている)
UC09a |済・一覧         | `pacman -Q`
UC09b |済・名前フィルタ | `pacman -Qs {regex}`
UC09c |済・手動属性     | `pacman -Qe`
UC09d |済・自動属性     | `pacman -Qd`
UC10  |済・詳細説明     | `pacman -Qi {pkgname}`
UC11  |済・構成一覧     | `pacman -Ql {pkgname}`
UC12  |済・構成逆引き   | `pacman -F {/path/to/file}`
UC13  |済・依存一覧     | `pactree {pkgname}`
UC14  |済・被依存一覧   | `pactree -r [-d {n}] {pkgname}`
UC15a |済・手動属性付与 | `pacman -D --asexplicit {pkgname}`
UC15b |済・自動属性付与 | `pacman -D --asdeps {pkgname}`
