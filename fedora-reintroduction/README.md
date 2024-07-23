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
設定は /etc/firewalld/zones/public.xml に書き込まれる。

    sudo firewall-cmd --permanent --add-port=3000/tcp
    sudo firewall-cmd --permanent --add-port=8000/tcp
    sudo firewall-cmd --permanent --add-port=8080/tcp
    sudo firewall-cmd --permanent --add-port=8888/tcp
    sudo firewall-cmd --reload

カメラ画像や音声をユーザーで扱えるようにするため、audio, render, video グループに自身を追加

    sudo usermod -a -G audio,render,video koron

ファームウェアをアップデート

    fwupdmgr get-devices    # デバイス確認
    fwupdmgr refresh        # 更新情報取得
    fwupdmgr get-updates    # 更新の有無を確認
    fwupdmgr update         # 更新適用

ROCm関連のパッケージをインストールする。
参考: <https://fedoraproject.org/wiki/SIGs/HC#Installation>

## GPUから応答がなくなる

rocminfoが機能しなくなる。
この時ディスプレイはおそらくスリープで消えていて、何をしても戻ってこない。
dmesgを確認するとamdgpuが死んでるっぽい。

lightdm-gtk-greeterがスクリーンセーバーに入ってしまったことが間接的な原因だと推測し、スリープしないように
/etc/lightdm/lightdm-gtk-gretter.conf に `screensaver-timeout=0` を書き足した。

再起動した直後のdmesgからamdgpuを抽出した。

* PSP = Platform Security Processor (PSP) DRM関連っぽい?
* VCN = Video Core Next 動画のハードウェアエンコーダー?

```
$ sudo dmesg | grep amdgpu
[    4.545961] [drm] amdgpu kernel modesetting enabled.
[    4.546090] amdgpu: Virtual CRAT table created for CPU
[    4.546108] amdgpu: Topology: Add CPU node
[    4.562749] amdgpu 0000:0a:00.0: No more image in the PCI ROM
[    4.562765] amdgpu 0000:0a:00.0: amdgpu: Fetched VBIOS from ROM BAR
[    4.562767] amdgpu: ATOM BIOS: 115-D754BP0-101
[    4.593512] amdgpu 0000:0a:00.0: amdgpu: CP RS64 enable
[    4.624360] amdgpu 0000:0a:00.0: [drm:jpeg_v4_0_early_init [amdgpu]] JPEG decode is enabled in VM mode
[    4.646605] amdgpu 0000:0a:00.0: vgaarb: deactivate vga console
[    4.646609] amdgpu 0000:0a:00.0: amdgpu: Trusted Memory Zone (TMZ) feature not supported
[    4.646670] amdgpu 0000:0a:00.0: amdgpu: VRAM: 16368M 0x0000008000000000 - 0x00000083FEFFFFFF (16368M used)
[    4.646673] amdgpu 0000:0a:00.0: amdgpu: GART: 512M 0x00007FFF00000000 - 0x00007FFF1FFFFFFF
[    4.646844] [drm] amdgpu: 16368M of VRAM memory ready
[    4.646847] [drm] amdgpu: 64355M of GTT memory ready.
[    4.648055] amdgpu 0000:0a:00.0: amdgpu: Will use PSP to load VCN firmware
[    4.705653] amdgpu 0000:0a:00.0: amdgpu: reserve 0x1300000 from 0x83fc000000 for PSP TMR
[    4.800172] amdgpu 0000:0a:00.0: amdgpu: RAS: optional ras ta ucode is not available
[    4.807753] amdgpu 0000:0a:00.0: amdgpu: RAP: optional rap ta ucode is not available
[    4.807756] amdgpu 0000:0a:00.0: amdgpu: SECUREDISPLAY: securedisplay ta ucode is not available
[    4.807787] amdgpu 0000:0a:00.0: amdgpu: smu driver if version = 0x00000035, smu fw if version = 0x00000040, smu fw program = 0, smu fw version = 0x00525b00 (82.91.0)
[    4.807791] amdgpu 0000:0a:00.0: amdgpu: SMU driver if version not matched
[    4.850356] amdgpu 0000:0a:00.0: amdgpu: SMU is initialized successfully!
[    5.037461] amdgpu 0000:0a:00.0: [drm:jpeg_v4_0_hw_init [amdgpu]] JPEG decode initialized successfully.
[    5.140633] amdgpu: HMM registered 16368MB device memory
[    5.142383] kfd kfd: amdgpu: Allocated 3969056 bytes on gart
[    5.142397] kfd kfd: amdgpu: Total number of KFD nodes to be created: 1
[    5.142445] amdgpu: Virtual CRAT table created for GPU
[    5.142591] amdgpu: Topology: Add dGPU node [0x7480:0x1002]
[    5.142594] kfd kfd: amdgpu: added device 1002:7480
[    5.142608] amdgpu 0000:0a:00.0: amdgpu: SE 2, SH per SE 2, CU per SH 8, active_cu_number 32
[    5.142613] amdgpu 0000:0a:00.0: amdgpu: ring gfx_0.0.0 uses VM inv eng 0 on hub 0
[    5.142616] amdgpu 0000:0a:00.0: amdgpu: ring comp_1.0.0 uses VM inv eng 1 on hub 0
[    5.142617] amdgpu 0000:0a:00.0: amdgpu: ring comp_1.1.0 uses VM inv eng 4 on hub 0
[    5.142619] amdgpu 0000:0a:00.0: amdgpu: ring comp_1.2.0 uses VM inv eng 6 on hub 0
[    5.142621] amdgpu 0000:0a:00.0: amdgpu: ring comp_1.3.0 uses VM inv eng 7 on hub 0
[    5.142622] amdgpu 0000:0a:00.0: amdgpu: ring comp_1.0.1 uses VM inv eng 8 on hub 0
[    5.142624] amdgpu 0000:0a:00.0: amdgpu: ring comp_1.1.1 uses VM inv eng 9 on hub 0
[    5.142626] amdgpu 0000:0a:00.0: amdgpu: ring comp_1.2.1 uses VM inv eng 10 on hub 0
[    5.142628] amdgpu 0000:0a:00.0: amdgpu: ring comp_1.3.1 uses VM inv eng 11 on hub 0
[    5.142629] amdgpu 0000:0a:00.0: amdgpu: ring sdma0 uses VM inv eng 12 on hub 0
[    5.142631] amdgpu 0000:0a:00.0: amdgpu: ring sdma1 uses VM inv eng 13 on hub 0
[    5.142633] amdgpu 0000:0a:00.0: amdgpu: ring vcn_unified_0 uses VM inv eng 0 on hub 8
[    5.142634] amdgpu 0000:0a:00.0: amdgpu: ring jpeg_dec uses VM inv eng 1 on hub 8
[    5.142636] amdgpu 0000:0a:00.0: amdgpu: ring mes_kiq_3.1.0 uses VM inv eng 14 on hub 0
[    5.146693] amdgpu 0000:0a:00.0: amdgpu: Using BACO for runtime pm
[    5.147397] [drm] Initialized amdgpu 3.57.0 20150101 for 0000:0a:00.0 on minor 1
[    5.154092] fbcon: amdgpudrmfb (fb0) is primary device
[    5.154097] amdgpu 0000:0a:00.0: [drm] fb0: amdgpudrmfb frame buffer device
[    6.533436] snd_hda_intel 0000:0a:00.1: bound 0000:0a:00.0 (ops amdgpu_dm_audio_component_bind_ops [amdgpu])
```

journalctl を確認する。
amdgpu カーネルモジュールが resume に失敗して停止し、X11が前提条件を満たせなくてコアダンプという流れのように見える。

* <https://wiki.archlinux.jp/index.php/AMDGPU>
* <https://docs.kernel.org/gpu/amdgpu/module-parameters.html>

カーネルモジュールパラメータの `amdgpu.runpm` あたりが怪しい。
以下のような設定を書き足して再起動…したが反映されてないぞ?

```console
$ cat /etc/modprobe.d/amdgpu.conf
# To disable Power Management. added 2024-07-20
options amdgpu runpm=0

$ restorecon -vF /etc/modprobe.d/amdgpu.conf

...

$ cat /sys/module/amdgpu/parameters/runpm
-1
```

カーネル起動引数にモジュールパラメータを足すことで対応した。
/etc/default/grub の `GRUB_CMDLINE_LINUX` に `amdgpu.runpm=0` を追加した後、
`grub2-mkconfig -o /boot/grub2/grub.cfg` で反映。

再起動すると `runpm` が 0 になり suspend しなくなり、問題が発生しなくなった。

## llama.cpp (with hipBLAS) のビルド

llama.cpp のビルドは以下の通り、何の問題も発生せずに完了した。
インストールしたパッケージはミニマルではない可能性がある。

```console
$ sudo dnf -y install hipblas-devel rocm-hip-devel rocblas-devel

$ git clone https://github.com/ggerganov/llama.cpp.git

$ cd llama.cpp
$ make -j 8 GGML_HIPBLAS=1
```

インストールは実行ファイルをコピーするだけに留めた。

```console
# mkdir -p /opt/llama.cpp/bin
# cp llama-* /opt/llama.cpp/bin
```
