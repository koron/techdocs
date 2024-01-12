# FreeBSDの新デフォルトMTA: DragonFly Mail Agent

FreeBSD 14.0 をインストールしたら /etc/mail/mailer.conf の中身が `/usr/libexec/dma` に変わっててMTAが変わったことを知った。

この機会に乗り換えてみるか調べ始めた。

## dma とは?

結果 dma とは DragonFly Mail Agent の略でソースコードはGitHubで管理されてることがわかった。

<https://github.com/corecode/dma/>

FreeBSDカーネルのソース /usr/src/contrib/dma/VERSION を見ると v0.13 であることがわかった。

```console
$ cat /usr/src/contrib/dma/VERSION
v0.13
```

ググってみたところこんな日本語記事が見つかった。

[dma(8) に置き換わった影響の回避法](https://qiita.com/je3kmz/items/038694a1b4ff7c640435)

> .forward 未実装って事じゃんが😱

[v0.13のTODO](https://github.com/corecode/dma/blob/v0.13/TODO)を見ると
確かに(今後サポート)すると書いてある。
っていうか9年前だな…いまさらよくsendmailを置き換えたな。

```
- unquote/handle quoted local recipients
- handle/use ESMTP extensions
- .forward support
- suggest way to run a queue flush on boot
```

トータル 4549 行らしいから、比較的手軽にメンテできるとは思う。

```
$ wc -l *.c *.h
     135 base64.c
     261 conf.c
     350 crypto.c
     122 dfcompat.c
     192 dma-mbox-create.c
     638 dma.c
     293 dns.c
     256 local.c
     523 mail.c
     690 net.c
     444 spool.c
     367 util.c
      24 dfcompat.h
     254 dma.h
    4549 total
```

## 有効化してみる

とりあえず有効化して試してみるか…

[公式ドキュメント: Chapter 31. Electronic Mail](https://docs.freebsd.org/en/books/handbook/mail/)

公式ドキュメントに従って、sendmailが起動しないよう&停止し、関連するperiodicを止めた。

/etc/mail/mailer.confを修正してdmaを使うように設定。

root→koronへメール送信テスト。
送信成功。

```console
# echo "This is just a small test message #1" | mail -s "Just a dma test #1" koron
```

koron→rootへメール送信テスト。
送信成功

```console
$ echo "This is just a small test message #4" | mail -s "Just a dma test #4" root
```

宛名にホスト名を付けて `koron@{MY HOST FQDN}` も成功。

宛名に別名ホスト `koron@{OTHER HOST FQDN}` を付けて送信、失敗、キューに滞留。
/var/log/maillog にログが出力されるのを確認。
/var/spool/dma にキューしたメールに対応するファイル(1メールにつきコンテンツとキュー情報の2ファイル)があるのを確認。
`/usr/libexec/dma -bp` でキューが見れるのを確認。

ソースコードを眺めて、キューの動作を確認。
初期リトライ300秒。
backoff は 1.5~2.5 倍で最大3時間。
5日間送れないとbounce

1つメールをキューすると1プロセス起動するっぽい。
つまりたくさんキューしちゃうとプロセスが溢れる。

あくまでもミニマルな運用のためのモノという感じ。
rootのcronが送るメールだけなら、確かにこれで充分な印象。

ただしcronの結果をメールじゃなくログに書くようにする設定方法もある。
実際にメールが送られなくなっても、いままでのsendmailを起動しておくよりはずっと良いか。

## その他資料

* <https://wiki.archlinux.org/title/Dma>
* [sendmail on FreeBSD の息の根を止める](https://mimosa-pudica.net/freebsd-sendmail.html)
