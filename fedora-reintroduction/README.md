# Fedora再導入記

浮いていたRyzen 9 3950XのPCに、Radeon RX 7600 XTを積んで機械学習系の何かをやってみたくなった。
ROCmやTritonなどAMDが向いている方向を実際に試してみておきたいという動機。
OSはLinux。
ディストリビューションは慣れたXubuntuやDebianも検討したが、本当に久しぶりにFedoraを使ってみることにした。
これはその導入に関する記録である。

## OSインストール

[Fedora Xfce 40](https://fedoraproject.org/ja/spins/xfce/download)からLive ISOを持ってきてUSBに書き込み。
ブートしてインストールという流れ。
初回は変にストレージパーティションをいじってしまったが、オートに任せたほうが良かろうということになった。
オートだとEFI System PertitionがVFAT、/bootがext4で切り出され、/homeと/がBtrfsとなり可変容量となり非常に都合が良い。
スワップは[Zswap](https://ja.wikipedia.org/wiki/Zswap)になる。

インストールが終わって再起動してログインしたら、XDGのユーザーフォルダー名を英語に変更。

    LANG=C xdg-user-dirs-update --force

パッケージを更新してリブートして不要になったパッケージの削除する。

    sudo dnf -y upgrade
    sudo reboot
    sudo dnf autoremove

sshd (SSHデーモン)は動いていないので有効化し、起動する。

    sudo systemctl enable sshd
    sudo systemctl start sshd
    systemctl status sshd

jupyter lab等をリモートから使うのに、いくつかのポートを解放する。
開けたのはとりあえず 3000, 8000, 8888 の3ポート。
必要になったら(必要なくなったら)順次足すか減らすか。

    sudo firewall-cmd --add-port=3000/tcp
    sudo firewall-cmd --add-port=8000/tcp
    sudo firewall-cmd --add-port=8888/tcp

カメラ画像や音声をユーザーで扱えるようにするため、audio & videoグループに自身を追加

    sudo usermod -a -G audio koron
    sudo usermod -a -G video koron

ファームウェアをアップデート

    fwupdmgr get-devices    # デバイス確認
    fwupdmgr refresh        # 更新情報取得
    fwupdmgr get-updates    # 更新の有無を確認
    fwupdmgr update         # 更新適用
