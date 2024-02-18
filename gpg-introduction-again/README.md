# GPG 再入門

Gnu Privacy Guardの略。
公開鍵暗号を用いて、メッセージの改ざんを検出する技術。
翻って改ざん防止の署名として機能する。

## 基礎知識

公開鍵暗号とは2つのペアとなる鍵を用意し、片方の鍵で暗号化したものがもう一方の鍵で復号できる暗号方式のこと。
片方の鍵を公開鍵として公開しておくことで、秘密鍵により暗号化された文章は秘密鍵を知ってる人により書かれたことを担保できる。
というのが基本的な仕組み。

大きな素数を用いたRSA暗号に始まり、楕円曲線暗号があたりが現在の主流だろうか。
GPGは2014年に楕円曲線暗号をサポートした。

署名への応用方法はおおよそ以下の手順による。
まず改ざんされてないことを保証したい文章本文のハッシュ値を計算。
秘密鍵を用いてハッシュ値を暗号化し署名とする。
本文と署名を送信。
受け取り手は署名を公開鍵で復号して、本文から計算したハッシュと比較。
同値であれば改ざんされていないと見做す。

GPGはPGPのオープンソース実装。

主鍵に対して副鍵を複数作成、利用できる。
どの副鍵で署名しても、主鍵に紐づいて検証できる。
副鍵毎に無効化できる。
このあたりは原理をちゃんと調べてない。

想定している使い方。
各副鍵を別々のマシンに配る。
マシン廃棄時・紛失時には副鍵を無効化して、公開している鍵情報を更新する。

公開する鍵情報は、メールアドレスに対してGPGのパブリックサーバー(?)にできる。
GitHub 内での署名目的であれば GitHub 自身に登録する。
設定場所は Settings → SSH and GPG keys → GPG keys 

参考までに以下は私のGitHubで確認できるGPG鍵の情報。
主鍵のIDが `13C36E2A4B75337B` で、3つの副鍵が登録されていることがわかる。
登録日は 2020/05/28 だったようだ。

```
Email address:  koron.kaoriya@gmail.com
Key ID: 13C36E2A4B75337B
Subkeys: E532735532FDE898 , B42B276F171874AF , E743618D98DD9BEB
Added on May 28, 2020
```

## 直近のゴール

前述の主鍵に対して…

1. 新しい副鍵を作成する
2. 使わなくなった副鍵を無効化する
3. 公開している鍵情報を更新する(パブリック&GitHub)
4. 新しい副鍵を、新PCにデプロイする
5. 鍵をバックアップする

## Hands on

とりあえず一覧は `gpg --list-keys`  

指紋を出したい場合は `gpg --fingerprint`

副鍵の指紋も出す場合は重ねる `gpg --fingerprint --fingerprint`

```
$ gpg --list-keys kaoriya
/home/koron/.gnupg/pubring.gpg
------------------------------
pub   rsa2048 2016-11-02 [SC]
      B6B0057BC2748D18A0E9D463A8DB2AF2A273719B
uid           [  究極  ] MURAOKA Taro (KoRoN) <koron.kaoriya@gmail.com>
sub   rsa2048 2016-11-02 [E]

pub   rsa4096 2020-05-23 [SC]
      1EAE763E7F10B43EE2122B8813C36E2A4B75337B
uid           [  究極  ] MURAOKA Taro (KoRoN) <koron.kaoriya@gmail.com>
sub   rsa4096 2020-05-23 [E]
sub   rsa4096 2020-05-28 [S]
sub   rsa4096 2020-05-28 [S]
```

`[SC]`, `[E]`, `[S]` の意味がわかんない。
調べたら以下のようだった。これ以外にも `A` 認証があるらしいが、手元の鍵にないので省略。

* `S` 署名 sign
* `C` 証明 certification
* `E` 暗号 encrypt

GitHub上での表示と Key ID が異なる件は `--keyid-format LONG` を指定すれば良い。
以降、2016-11-02の鍵は古いので無視する。あとで無効化しよう。

```
$ gpg --list-keys --keyid-format LONG 13C36E2A4B75337B
pub   rsa4096/13C36E2A4B75337B 2020-05-23 [SC]
      1EAE763E7F10B43EE2122B8813C36E2A4B75337B
uid                 [  究極  ] MURAOKA Taro (KoRoN) <koron.kaoriya@gmail.com>
sub   rsa4096/E532735532FDE898 2020-05-23 [E]
sub   rsa4096/B42B276F171874AF 2020-05-28 [S]
sub   rsa4096/E743618D98DD9BEB 2020-05-28 [S]
```

GitHub に登録する際の形式は `--armor --export {鍵ID}` で標準出力へ出力される。
クリップボードなどに流してブラウザに持って行けば良い。
二重登録はできない。

```
$ gpg --armor --export 13C36E2A4B75337B | clip
```



## (TODO)

## 参考リンク

* [About commit signature verification](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification)
* [GnuPG で遊ぶ - 暗号化してみる (2013-07-31)](https://blog.eiel.info/blog/2013/07/31/gpg/)
* [GnuPG チートシート（鍵作成から失効まで） (2022-03-27)](https://text.baldanders.info/openpgp/gnupg-cheat-sheet/)
* [GnuPG チートシート（簡易版） (2020-09-20)](https://zenn.dev/spiegel/articles/20200920-gnupg-cheat-sheet)
