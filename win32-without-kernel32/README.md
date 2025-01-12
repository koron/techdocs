# kernel32.dllを使わないプログラムの検証

これだけのプログラム(program.c)を用意して…

```c
void mainCRTStartup(void) {
}
```

以下のようにコンパイル&リンクして

```console
$ cl -c program.c

$ link program.obj -entry:mainCRTStartup -subsystem:console

$ ls -l
合計 6
-rw-r--r-- 1 koron None   32  1月 12 13:15 program.c
-rwxr-xr-x 1 koron None 1536  1月 12 13:16 program.exe*
-rw-r--r-- 1 koron None  633  1月 12 13:18 program.obj
```

実行しても何もしないプログラムができるんだけど、kernel32.dll どころか一切のDLLをロードしないEXEになってる。

```console
$ dumpbin -imports program.exe
Microsoft (R) COFF/PE Dumper Version 14.34.31937.0
Copyright (C) Microsoft Corporation.  All rights reserved.


Dump of file program.exe

File Type: EXECUTABLE IMAGE

  Summary

        1000 .rdata
        1000 .text
```

リンカーに `-entry` オプションでエントリーポイントを指定して、
`-subsystem` でサブシステムを明示してあげている。

program.exe をエクスプローラーからダブルクリックで実行すると、
一瞬だけコンソール (Windows Terminal) が表示されるのが確認できる。
これはサブシステムに `console` を指定したことによる。

しかしこのコンソールに `Hello World` 等を表示するには
[GetStdHandle](https://learn.microsoft.com/ja-jp/windows/console/getstdhandle) かCreateFileでコンソールの出力ハンドルを取得し、
WriteFile等でメッセージを書き込む必要があるので
その時点で kernel32.dll が必要になる。

コンソールを割り当てるのはユーザープロセス外の、
OSの仕事であることがわかる。
