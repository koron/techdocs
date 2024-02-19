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
もちろん本文全体を秘密鍵で暗号化するのでも、署名として機能する。

GPGはPGPのオープンソース実装。

主鍵に対して副鍵を複数作成、利用できる。
どの副鍵で署名しても、主鍵に紐づいて検証できる。
副鍵毎に無効化できる。
このあたりは原理をちゃんと調べてない。

### Git 及び GitHubでの署名

[About commit signature verification](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification)を見れば基本的なことは書いてある。

git 側は `config` で `user.name`, `user.email`, `user.signingkey` に値を設定し、`gpg.program=gpg` を設定すれば `-S` オプションで署名できるようになる。
`commit.gpgsign=true` や `tag.gpgsign=true` を設定すれば、それぞれの操作にGPG署名を必須にできる。

通常であればこれらの git の設定はグローバル config に入れれば十分である。

## 私のユースケース

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

加えてレポジトリごとに複数(2種類)の署名を使い分けている。
使い分け方は .gitconfig からの抜粋の通り。
commitとtagサブコマンドに署名を必須にしており、
identityエイリアスで適切な署名をレポジトリごとに設定している。
例: `git identity private`


```ini
[user "private"]
	name = MURAOKA Taro
	email = koron.kaoriya@gmail.com
	signingkey = 13C36E2A4B75337B

[user "work"]
	name = Taro Muraoka
	email = taro.muraoka@work.example.com
	signingkey = XXXXXXXXXXXXXXXX

[gpg]
	program = gpg

[commit]
	gpgsign = true

[tag]
	gpgsign = true

[alias]
	identity = "! git config user.name \"$(git config user.$1.name)\"; git config user.email \"$(git config user.$1.email)\"; git config user.signingkey \"$(git config user.$1.signingkey)\"; :"
```

この .gitconfig は各マシン間で共有しており、
配った先で signingkey を副鍵に合わせて書き換えるということはしたくない。

## 直近のゴール

前述の主鍵に対して…

1. 新しい副鍵を作成する ✓
2. 使わなくなった副鍵を無効化する
3. 公開している鍵情報を更新する(パブリック&GitHub) ✓
4. 新しい副鍵を、新PCにデプロイする ✓
5. 鍵をバックアップする ✓

追加の課題

* 親機で署名に使われるsubkeyの選択方法 ✓
* 過去機で使ってたsubkeyの特定 ✓

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

* `S` 署名 sign : 署名に使える
* `C` 証明 certification : 副鍵の作成に使える
* `E` 暗号 encrypt : 暗号化に使える

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

```
$ gpg --armor --export 13C36E2A4B75337B | clip
```

適当に主鍵を作ってみて、消した。
秘密鍵を消してから公開鍵を消す。
秘密鍵を消す際にはパスフレーズが聞かれる。
対応する秘密鍵のない公開鍵は、他人の鍵同様の扱い。

```
$ gpg --delete-secret-key 85A760630D29548C
$ gpg --delete-key 85A760630D29548C
```

副鍵を作る。
`gpg --edit-key 13C36E2A4B75337B` で鍵の編集モードに入る。
`addkey` で副鍵の編集モードに入り、ポチポチやってたらなんかできた。
この副鍵では楕円曲線暗号を採用。

```
$ gpg --list-secret-keys --keyid-format=long 13C36E2A4B75337B
sec   rsa4096/13C36E2A4B75337B 2020-05-23 [SC]
      1EAE763E7F10B43EE2122B8813C36E2A4B75337B
uid                 [  究極  ] MURAOKA Taro (KoRoN) <koron.kaoriya@gmail.com>
ssb   rsa4096/E532735532FDE898 2020-05-23 [E]
ssb   rsa4096/B42B276F171874AF 2020-05-28 [S]
ssb   rsa4096/E743618D98DD9BEB 2020-05-28 [S]
ssb   ed25519/F1A0E61C700E0663 2024-02-18 [S]
```

とりあえず GitHub に長しこんでみる。
主鍵が同じものの二重登録はできないので、
更新時には古いものを消してから新しいものを再度追加する。

```
$ gpg --armor --export 13C36E2A4B75337B | clip
```

更新後のGitHub上での表示はこうなった。
Subkeysに `F1A0E61C700E0663` が増えて、Added onが作業当日の日付になった。
過去の Verified コミットマークも維持されてる。

```
Email address:  koron.kaoriya@gmail.com
Key ID: 13C36E2A4B75337B
Subkeys: E532735532FDE898 , B42B276F171874AF , E743618D98DD9BEB , F1A0E61C700E0663
Added on Feb 18, 2024
```

副鍵の秘密鍵をファイルにエクスポート。
出力したファイル F1A0E61C700E0663.asc はノートPCへコピー。

```
gpg --armor --export-secret-key F1A0E61C700E0663 > F1A0E61C700E0663.asc
```

親機で主鍵のID 13C36E2A4B75337B で署名すると、
最後に作った副鍵が使われるのかも。ちょっと厄介ね。

`export-secret-subkeys` じゃないとダメらしい。
鍵IDに `!` も付けなきゃダメ。
付けないと全ての副鍵がエクスポートされる。
結果、exportコマンドはこうなった。

```
gpg --armor --export-secret-subkey F1A0E61C700E0663! > F1A0E61C700E0663.asc
```

インポートは非常に単純。
これだけで `signingkey=13C36E2A4B75337B` で F1A0E61C700E0663 を使ってサインできるようになる。

```
gpg --import F1A0E61C700E0663.asc
```

親機で F1A0E61C700E0663 を追加する前はgitからは E532735532FDE898 が使われていた。
戻すにはどうしたら良いか?

フロントエンドの問題かもなので、以下のコマンドでresetしてみる。
が、関係なさそう。

```
$ echo RELOADAGENT | gpg-connect-agent
```

E743618D98DD9BEB は生きてはいるが使ってないPCにインストールされていた。
B42B276F171874AF は処分するノートPCに入っていた。
失効させられるかも。

鍵のバックアップは ~/.gnupg を丸々 `tar cvf backup.tar.gz -C ~ .gnupg` でアーカイブ化。

署名に利用される副鍵は、リスト中の利用可能な最後の `S` 付き秘密鍵であるようだった。
`gpg --list-secret-keys --with-keygrip` でリストを取ると秘密鍵のファイル名が判明する。
判明したファイルを名前変更し利用不可にすることで、意図しない秘密鍵を利用できないようにした。
3つの署名鍵が `ssb#` となっており、秘密鍵のファイルにアクセスできないため、利用できないことを示している。
この状態では一番最初の `S` である `13C36E2A4B75337B` が使われる。

```console
$ gpg --list-secret-keys --keyid-format=long 13C36E2A4B75337B
sec   rsa4096/13C36E2A4B75337B 2020-05-23 [SC]
      1EAE763E7F10B43EE2122B8813C36E2A4B75337B
uid                 [  究極  ] MURAOKA Taro (KoRoN) <koron.kaoriya@gmail.com>
ssb   rsa4096/E532735532FDE898 2020-05-23 [E]
ssb#  rsa4096/B42B276F171874AF 2020-05-28 [S]
ssb#  rsa4096/E743618D98DD9BEB 2020-05-28 [S]
ssb#  ed25519/F1A0E61C700E0663 2024-02-18 [S]
```

## まとめ

以下に今後も使いそうな GPG の操作をまとめておく。

1.  秘密鍵の一覧

    ```console
    $ gpg --list-secret-keys --keyid-format=long
    ```

    秘密鍵ファイルの操作が必要な場合は `--with-keygrip` を付ける。

    ```console
    $ gpg --list-secret-keys --keyid-format=long --with-keygrip
    ```

    どちらも引数を足すことで、表示する鍵を絞れる。

2.  公開鍵のエクスポート(用途例: GitHubへの登録)

    ```console
    $ gpg --export --armor {主鍵のID}
    ```

    標準出力に出力されるのでファイルに書きだすなり、
    クリップボードへ転送するなりするとよい。

3. 副鍵の作成および失効

    主鍵の編集モードに入る。

    ```console
    $ gpg --edit-key {主鍵のID}
    ```

    副鍵の追加は `addkey` で質問に答える形で作れる。
    作った後は `save` で保存を忘れずに。

    失効は `key {N}` で失効させたい鍵を選択し、
    `revkey` で失効。
    `save` が要るかは、まだ試してないので不明。

    どちらの場合も操作後に 2 の手順で公開鍵のエクスポートと
    各サービスへの登録・更新操作が要る。

4. 副鍵の配布

    副鍵の秘密鍵をファイルへエクスポート。
    副鍵のIDの後ろに `!` があることに注意。
    忘れると副鍵の主鍵に紐づくすべての秘密鍵がエクスポートされてしまう。

    ```console
    gpg --export-secret-subkey --armor {副鍵のID}! > subkey.asc
    ```

    できたファイルを安全な手段で配布先PCへ転送しインポート。

    ```console
    gpg --import subkey.asc
    ```

5. 副鍵を削除

    配布先から副鍵を消すには `gpg --delete-secret-key {鍵ID}`

    PCを廃棄するのであればディレクトリごと消すのも可 `rm -rf ~/.gnupg`

6. 副鍵を署名に利用しないようにする

    主鍵に複数の副鍵がある場合に特定の副鍵を使わないようにする方法。
    複数の署名用の鍵がある場合、特に指定しない場合にGPGはリスト中の利用可能な最後の鍵を使うようだ。

    まず `gpg --list-secret-keys --with-keygrip` で利用不可にしたい副鍵の Keygrip を特定する。

    ~/.gnupg/private-keys-v1.d/ ディレクトリ下にある `{Keygrip}.key` をリネームする。
    次の例ではファイル名の末尾に `~` を追加することで、GPGから利用できないようにしている。
    `XXX...` は Keygrip に置き換えること。

    ```console
    $ mv ~/.gnupg/private-keys-v1.d/XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX.key{,~}
    ```

    戻す時はその逆を行えばよい。

    ```console
    $ mv ~/.gnupg/private-keys-v1.d/XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX.key{~,}
    ```

以上。より基本的なことは
[About commit signature verification](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification)
が詳しいので、そちらも参照すること。
より発展的なことは、参考リンクを参照すること。

## 参考リンク

* [About commit signature verification](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification)
* [GnuPG で遊ぶ - 暗号化してみる (2013-07-31)](https://blog.eiel.info/blog/2013/07/31/gpg/)
* [GnuPG チートシート（鍵作成から失効まで） (2022-03-27)](https://text.baldanders.info/openpgp/gnupg-cheat-sheet/)
* [GnuPG チートシート（簡易版） (2020-09-20)](https://zenn.dev/spiegel/articles/20200920-gnupg-cheat-sheet)
* [gpg のはなし](https://gist.github.com/hatsusato/1d5f0267bc9d02bb24c60bd7acc5a59a)
