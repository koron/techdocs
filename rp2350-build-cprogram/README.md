# RP2350 用のC言語プログラムをビルドする

環境はWindows + MSYS2 + MinGW64(変則)

## picotool をビルドする

picotool はブートモードのRP2350の情報を表示したり、実効バイナリ(`*.uf2`)の内容を表示したりできるツール。
RP2040用のはビルドしてインストールしていたが、新たにRP2350に対応させるために再コンパイル&インストールする必要があった。

1. 最新の[raspberry/picotool](https://github.com/raspberrypi/picotool) をクローン

        $ git clone https://github.com/raspberrypi/picotool.git
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

    Ninja でビルドすることもできそうなので、その場合は以下のコマンドで生成する。

        $ cmake -B build -DCMAKE_INSTALL_PREFIX=/d/msys64/usr/local

3. ビルドする

        $ cmake --build build

    実は cmake にはビルドを実行する機能もあるぞ

4. インストールする

        $ cmake --install build

    実は cmake にはインストールする機能もある。
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

1. pico-sdk をクローン

    とりあえずビルドしたいだけなら、以下のコマンドで shallow クローンすれば充分。
    クローンした先のパスを環境変数 `PICO_SDK_PATH` へ設定する。

        $ git clone --depth 1 https://github.com/raspberrypi/pico-sdk.git -b 2.0.0
        $ cd pico-sdk
        $ git submodule update --init --recursive --depth=1
        $ export PICO_SDK_PATH="$(pwd)"

    完全な履歴を取得しメインブランチに追従する場合は、
    以下のように短いコマンドに変わるが実行にかかる時間は長くなるので、
    そこそこ覚悟が必要。

        $ git clone --recursive https://github.com/raspberrypi/pico-sdk.git
        $ export PICO_SDK_PATH="$(pwd)"

2. pico-examples をクローン

    pico-sdk とは別の場所にクローンする。

        $ git clone https://github.com/raspberrypi/pico-examples.git

    カレントディレクトリを pico-examples にすることを忘れずに。

        $ cd pico-examples

3. ビルドファイルを生成する

    *   ビルドに使うディレクトリは ./build/rp2350
    *   `PICO_PLATFORM=rp2350` を定義することでビルドターゲットをRP2350にする
    *   ビルドにはmsysのmakeを使う

    以上のことを考慮してビルドファイルの生成コマンドは以下のようにする。

        $ cmake -B ./build/rp2350 -DPICO_PLATFORM=rp2350 -G "MSYS Makefiles"

    MSYS Makefiles ではなく Ninja を使う場合は以下のようにする。
    Ninja のほうが並列にビルドが進み、速い。

        $ cmake -B ./build/ninja-rp2350 -DPICO_PLATFORM=rp2350

4. ビルドする

    ビルドにMSYS Makefilesを使う場合、サブディレクトリだけをコンパイルするには以下のようにする。
    ディレクトリを指定することで特定のプログラムだけをコンパイルする。

        $ cmake --build ./build/rp2350/hello_world/serial

    ビルドにNinjaを使っている場合は `-t` でターゲット指定を追加することで、特定のプログラムだけをコンパイルできる。

        $ cmake --build ./build/ninja-rp2350 -t hello_world/serial/all

    出力ファイルは `build/rp2350/hello_world/serial/hello_serial.uf2` となる。
    `file` コマンドや `picotool` コマンドで中身を確認できる。

        $ file build/rp2350/hello_world/serial/hello_serial.uf2
        build/rp2350/hello_world/serial/hello_serial.uf2: UF2 firmware image, family 0xe48bff57, address 0x10ffff00, 2 total blocks

        $ picotool info -a build/rp2350/hello_world/serial/hello_serial.uf2

        File build/rp2350/hello_world/serial/hello_serial.uf2:

        Program Information
         name:          hello_serial
         web site:      https://github.com/raspberrypi/pico-examples/tree/HEAD/hello_world/serial
         features:      UART stdin / stdout
         binary start:  0x10000000
         binary end:    0x10001f20
         target chip:   RP2350
         image type:    ARM Secure

        Fixed Pin Information
         0:  UART0 TX
         1:  UART0 RX

        Build Information
         sdk version:       2.0.0
         pico_board:        pico2
         build date:        Sep  1 2024
         build attributes:  Release

5. 実機にインストールする

    自分の環境ではEドライブにPico 2デバイスがマッピングされているので、
    インストール(≒ただのコピー)のコマンドは以下のようになる。

        $ cp build/rp2350/hello_world/serial/hello_serial.uf2 /e/

## TIPS

*   蛇足だが `cmake -B {build_dir}` もしくは `cmake --build {build_dir}` に指定するビルドディレクトリ `{build_dir}` は任意の場所に変更して良い
*   picotoolのビルドには MSYS2 camke ではなく MinGW64 cmake を使う
*   `PICO_PLATFORM=rp2350` を指定することで RP2350 用にビルドできる

    ビルドファイル生成時じゃなくて、ビルド時に指定しても機能するのか? (機能しなかった)

*   `-G` を指定しないとmakeじゃなくてninjaになる

    Ninjaのほうがビルドの時間が短縮できる。
    またNinjaでサブディレクトリだけをビルドする場合は
    次のように `-t {subdir}/all` を付ける。

        $ cmake --build build -t {subdir}/all

    参考: https://cmake.org/cmake/help/latest/generator/Ninja.html
