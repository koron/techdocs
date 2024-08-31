# RP2350 用のC言語プログラムをビルドする

環境はWindows + MSYS2 + MinGW64(変則)

## picotool をビルドする

picotool はブートモードのRP2350の情報を表示したり、実効バイナリ(`*.uf2`)の内容を表示したりできるツール。
RP2040用のはビルドしてインストールしていたが、新たにRP2350に対応させるために再コンパイル&インストールする必要があった。

1. 最新の[raspberry/picotool](https://github.com/raspberrypi/picotool) をチェックアウト

        $ git clone https://github.com/raspberrypi/picotool.git
        $ git submodule update
        $ cd picotool

    ビルドに必要な情報は[README](https://github.com/raspberrypi/picotool?tab=readme-ov-file#for-windows-with-mingw-in-msys2)に書かれてる。
    mingwの toolchain, cmake, libusb が必要なのであらかじめインストールする必要がある。
    ドキュメントでは `$MINGW_PACKAGE_PREFIX` を用いているが、自分の環境は変則であるため直接 `mingw-w64-x86_64` を指定している。

2. ビルドファイルを生成する

    *   ビルドに使うディレクトリは ./build
    *   ビルドにはmsysのmakeを使う
    *   インストール先を `/usr/local` (`D:\msys64\usr\local`) にする

    そのため設定はオプションは以下のようになる。

        $ cmake -B build -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=/d/msys64/usr/local

3. ビルドする

        $ cmake --build build

    実は cmake にはビルドを実行する機能もあるぞ

4. インストールする

        $ cmake --install build

    実は cmake にはインストールする機能もあるぞ。
    インストールされるファイルは以下の通り。

        bin/picotool.exe
        lib/cmake/picotool/picotoolTargets.cmake
        lib/cmake/picotool/picotoolTargets-release.cmake
        lib/cmake/picotool/picotoolConfig.cmake
        lib/cmake/picotool/picotoolConfigVersion.cmake
        share/picotool/rp2350_otp_contents.json
        share/picotool/xip_ram_perms.elf

## hello world (serial)をビルド

[pico-examples](https://github.com/raspberrypi/pico-examples) の `hello_world/serial` をビルドする。

1. レポジトリをチェックアウト
2. ビルドファイルを生成する

    *   ビルドに使うディレクトリは ./build/rp2350
    *   `PICO_PLATFORM=rp2350` を定義することでビルドターゲットをRP2350にする
    *   ビルドにはmsysのmakeを使う

        $ cmake -B ./build/rp2350 -DPICO_PLATFORM=rp2350 -G "MSYS Makefiles"

3. ビルドする

        $ cmake --build ./build/hello_world/serial

    ディレクトリを指定することで特定のプログラムだけをコンパイルする。
    出力ファイルは `build/rp2350/hello_world/serial/hello_serial.uf2`

        $ file build/rp2350/hello_world/serial/hello_serial.uf2
        build/rp2350/hello_world/serial/hello_serial.uf2: UF2 firmware image, family 0xe48bff57, address 0x10ffff00, 2 total blocks

4. 実機にインストールする

        $ cp build/rp2350/hello_world/serial/hello_serial.uf2 /e/

    自分の環境はEドライブにPico 2デバイスがマッピングされている。

## TIPS

*   picotoolのビルドには MSYS2 camke ではなく MinGW64 cmake を使う
*   `PICO_PLATFORM=rp2350` を指定することで RP2350 用にビルドできる

    ビルドファイル生成時じゃなくて、ビルド時に指定しても機能するのか? (機能しなかった)
*   `-G` を指定しないとmakeじゃなくてninjaになる

    Ninjaを使うとプロジェクトの一部がコンパイルできない。
