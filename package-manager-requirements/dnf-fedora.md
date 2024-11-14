# dnf (Fedora)

ユースケース毎のコマンド

ID    | ユースケース    | コマンド
:----:|-----------------|--------------------------------------------------------------
UC01  |インストール     | `dnf install {pkgname}`
UC02  |アンインストール | `dnf remove {pkgname}`
UC03  |クリーンナップ   | `dnf autoremove`
UC04  |パッケージ更新   | `dnf upgrade`
UC05  |レポジトリの同期 | `dnf --refresh upgrade`
UC06  |未・一覧         | `dnf list --available [pattern]`
UC07  |未・詳細説明     | `dnf info [--available] {pkgname}`
UC08  |未・依存一覧     | `dnf repoquery --requires {pkgname}`
UC09a |済・一覧         | `dnf list --installed`
UC09b |済・名前フィルタ | `dnf list --installed {glob}`
UC09c |済・手動属性     | `dnf repoquery --userinstalled`
UC09d |済・自動属性     | `dnf repoquery --installed --qf='%{name}-%{evr}.%{arch} (%{reason})\n' \| grep -v '(User)'`
UC10  |済・詳細説明     | `dnf info [--installed] {pkgname}`
UC11  |済・構成一覧     | `dnf repoquery [--installed] -l {pkgname}`
UC12  |済・構成逆引き   | `dnf repoquery [--installed] -f {/path/to/file}`
UC13  |済・依存一覧     | `dnf repoquery --requires {pkgname}`
UC14  |済・被依存一覧   | `dnf repoquery --whatrequires {pkgname} --installed`
UC15a |済・手動属性付与 | `dnf mark install {pkgname}`
UC15b |済・自動属性付与 | `dnf mark remove {pkgname}`
