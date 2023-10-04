# Memo for how GOROOT determined

以下はGo 1.21.1 Windowsでの調査結果

Goのプログラムの中では `GOROOT` は `runtime.GOROOT()` で取得できる。
この時、環境変数GOROOTが設定されていればそれが取得できる。
設定されていない場合に何が取得できるのかが問題。

実際に `runtime.GOROOT()` 返す値は `runtime.defaultGOROOT` であり、
これはリンク時に決定される。

https://github.com/golang/go/blob/2c1e5b05fe39fc5e6c730dd60e82946b8e67c6ba/src/runtime/extern.go#L304-L315

```go
var defaultGOROOT string // set by cmd/link

// GOROOT returns the root of the Go tree. It uses the
// GOROOT environment variable, if set at process start,
// or else the root used during the Go build.
func GOROOT() string {
	s := gogetenv("GOROOT")
	if s != "" {
		return s
	}
	return defaultGOROOT
}
```

そこからリンカーのコードを辿ると、
ビルド時の `buildcfg.GOROOT` 由来であることがわかる。

https://github.com/golang/go/blob/2c1e5b05fe39fc5e6c730dd60e82946b8e67c6ba/src/cmd/link/internal/ld/main.go#L129-L135

```go
if final := gorootFinal(); final == "$GOROOT" {
    // cmd/go sets GOROOT_FINAL to the dummy value "$GOROOT" when -trimpath is set,
    // but runtime.GOROOT() should return the empty string, not a bogus value.
    // (See https://go.dev/issue/51461.)
} else {
    addstrdata1(ctxt, "runtime.defaultGOROOT="+final)
}
```

https://github.com/golang/go/blob/2c1e5b05fe39fc5e6c730dd60e82946b8e67c6ba/src/cmd/link/internal/ld/pcln.go#L786-L792

```go
func gorootFinal() string {
	root := buildcfg.GOROOT
	if final := os.Getenv("GOROOT_FINAL"); final != "" {
		root = final
	}
	return root
}
```

んじゃ `buildcfg.GOROOT` がどうなってるのかを調べると、
`runtime.GOROOT()` に戻ってきてしまう。
ただしこれはコンパイラ&リンカのバイナリをビルドしたときのGOROOTになっていることに注意が必要。

https://github.com/golang/go/blob/2c1e5b05fe39fc5e6c730dd60e82946b8e67c6ba/src/internal/buildcfg/cfg.go#L24

```
GOROOT   = runtime.GOROOT() // cached for efficiency
```

もとの`runtime.GOROOT()`に戻ると、1行目の `gogetenv("GOROOT")` が怪しい。
仮にそいつが怪しくないとすると、
defaultGOROOTにはリンカをビルドしたときのGOROOTが入っていることになり
別のプログラムをビルドする際の`runtime.GOROOT()` は
常にそれを返すことになってしまう。
即ち、配布されたコンパイラでビルドしたプログラムは、
実際にコンパイラが配置されているのとは異なるGOROOTを返すことになり、
辻褄が合わない。

なお runtime.defaultGOROOT は runtime.GOROOT() を利用しているプログラムにしか埋め込まれないようだ。

(TODO: defaultGOROOT の値を直接確かめたい)

ビルド時に`GOROOT_FINAL`を設定するとそれが埋め込まれているのはほぼ確実。

```console
$ GOROOT_FINAL=foo go run ./test1.go
runtime.GOROOT=foo
build.Default.GOROOT=foo

$ GOROOT_FINAL=bar go run ./test1.go
runtime.GOROOT=bar
build.Default.GOROOT=bar

$ GOROOT_FINAL=baz go build ./test1.go

$ ./test1.exe
runtime.GOROOT=baz
build.Default.GOROOT=baz
```

やはり未設定時はbuildcfg.GOROOTの問題になりそう。

どこかでGOROOTの値を、補正する構造を見た記憶があるのだが…

cmd/go/internal/cfg/cfg.go 内のSetGOROOT() 関連か。
ここでBuildContextはgo/build.DefaultContext相当らしい

https://github.com/golang/go/blob/2c1e5b05fe39fc5e6c730dd60e82946b8e67c6ba/src/cmd/go/internal/cfg/cfg.go#L194-L195

```go
func SetGOROOT(goroot string, isTestGo bool) {
    BuildContext.GOROOT = goroot
```

https://github.com/golang/go/blob/2c1e5b05fe39fc5e6c730dd60e82946b8e67c6ba/src/cmd/go/internal/cfg/cfg.go#L183-L184

```go
func init() {
    SetGOROOT(Getenv("GOROOT"), false)
```

このGetenvはcfg自前で、環境変数が設定されていればソレを、
されてなければenvCacheを利用する。
envCacheは一度だけinitEnvCacheで初期化される。

https://github.com/golang/go/blob/2c1e5b05fe39fc5e6c730dd60e82946b8e67c6ba/src/cmd/go/internal/cfg/cfg.go#L371-L386

```go
func Getenv(key string) string {
    if !CanGetenv(key) {
        switch key {
        case "CGO_TEST_ALLOW", "CGO_TEST_DISALLOW", "CGO_test_ALLOW", "CGO_test_DISALLOW":
            // used by internal/work/security_test.go; allow
        default:
            panic("internal error: invalid Getenv " + key)
        }
    }
    val := os.Getenv(key)
    if val != "" {
        return val
    }
    envCache.once.Do(initEnvCache)
    return envCache.m[key]
}
```

initEnvCacheではGOROOTだけ特別扱いでfindGOROOT()で探す。

https://github.com/golang/go/blob/2c1e5b05fe39fc5e6c730dd60e82946b8e67c6ba/src/cmd/go/internal/cfg/cfg.go#L306-L321

findGOROOT()ではgo.exeの位置からGOROOTを推定する。
go.exeの1つ上、もしくは2つ上のディレクトリがisGOROOT()を満たすならGOROOTとする。
isGOROOT()はpkg/toolsディレクトリの有無で判定に代えている。

https://github.com/golang/go/blob/2c1e5b05fe39fc5e6c730dd60e82946b8e67c6ba/src/cmd/go/internal/cfg/cfg.go#L504-L533

test2.go で実験したところ os.Executable はPATHでの表記+実行ファイルという文字列を返す。
つまり go.exe が PATH に D:\go\current\bin (小文字)と書かれた場所で見つかれば
os.Executable() は D:\go\current\bin\go.exe を返し、ビルド時のGOROOTは D:\go\current になる。
またPATHに D:\Go\current\bin (大文字小文字混在)と書かれていれば、
os.Executable() は D:\Go\current\bin\go.exe を返し、ビルド時のGOROOTは D:\Go\current になる。

ここまでくると go.exe 内で推定された GOROOT がリンカーに正しく渡ってることをも確認したい。
両者は別のコマンドなので、Setenvなどで子プロセスの起動に伝えてるはず。

https://github.com/golang/go/blob/2c1e5b05fe39fc5e6c730dd60e82946b8e67c6ba/src/cmd/go/internal/toolchain/exec.go#L22-L27

execGoToolchain() というのがあった。
ここのdirに渡ってくるはずなので逆引きしていく。

```go
func execGoToolchain(gotoolchain, dir, exe string) {
        os.Setenv(targetEnv, gotoolchain)
        if dir == "" {
                os.Unsetenv("GOROOT")
        } else {
                os.Setenv("GOROOT", dir)
```

そっちじゃなくて internal/work/exec.go の runOut() の envのほうかも?

ちょっと追うのが面倒になったので、いったん終了。

go/main.go で `os.Setenv()` を読んで設定してた。
envcmd.MkEnv()で `go env` の出力相当をとってきて、
実際の値と異なるものだけ `os.Setenv()` している。
go起動時は環境変数GOROOTは設定されていないが
cfg.GOROOTには設定されていて値が異なるため `os.Setenv()` することになる。
