# systemd-resolved

systemd-resolved で dnsmasq に問い合わせたら期待してない結果が出た。
その原因を調べて迂回策を設定した。

## 状況

* 自分で所有しているドメイン(以下 example.com とする)を使っている
* グローバルDNSの設定
    * ドメインの下に3つのホスト foo, bar, baz がぶら下がっている
    * foo というホストはAレコードを持っている
    * bar と baz は foo.example.com への CNAME 設定
* ローカルDNSの設定
    * 自宅LAN内では dnsmasq を使って bar, baz のIPをLAN内でのIPにしている
    * WindowsやAndroid では bar.example.com, baz.example.com で期待通りのIPを見れる
    * Xubuntu では両FQDNに対して foo.example.com のグローバルIPが返ってくる

つまり Xubuntu の systemd-resolved だけが何かおかしい振舞いをする

## 調査

resolvectl で systemd-resolved に問い合わせる。

```console
$ resolvectl query bar.example.com
```

これは CNAME の foo.example.com になり
dnsmasq には foo.example.com のエントリがないため
グローバルのIPで解決していた。

だから dnsmasq で foo.example.com へローカルのIPを返すようにすれば一時しのぎの解決策にはなる。
しかしそもそもなんでCNAMEが出てくるか不明。

dnsmasq に `log-queries` と `log-facility=/var/log/dnsmasq.conf` を設定して再度問合せ。
結果 bar.example.com に対して AAAA を問い合わせたのちに A を問い合わせてることがわかった。

AAAA についてはグローバルもローカルも設定していない。
dnsmasq はグローバルにフォワードし CNAME が含まれた NS レコード相当が返ってくる。

systemd-resolved は A レコードも得ているが、AAAA の問合せで返ってきた CNAME を結果的に優先し
foo.example.com の A レコードを bar.example.com のものとしてレポートしてきていた。

## 解決策

dnsmasq が所有しているドメイン example.com についての未知の情報を上位のDNSサーバーに転送してしまうのが問題。
example.com ゾーンの権威として振る舞わせれば、転送は起こらない良いはず。

`man dnsmasq` を眺めていると `auth-server` と `auth-zone` という設定を発見。
試してみたところ以下の設定で AAAA の転送をしなくなった。

```
auth-server=example.com
auth-zone=example.com
```

## まとめ

* systemd-resolved は AAAA と A を問合せ、Aの結果を捨てる場合がある
    * AAAA で問い合わせて出てきた CNAME を A で解決した上に、直で引ける A より優先するのは若干おかしいと思う
* dnsmasq はゾーンの権威として振る舞える
    * /etc/hosts でお手軽にエントリを管理できるが、ちゃんと設定しないとダメな場合もある
