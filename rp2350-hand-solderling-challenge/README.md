# RP2350A手はんだキットへの挑戦記

74thさんによるキットを手に入れたので、その記録

*   BOOTHの販売ページ: <https://booth.pm/ja/items/6483839>
*   レポジトリ: <https://github.com/74th/rp2040-dev-board/tree/main/rp2350a-full>
*   組み立てレポート: <https://74th.hateblo.jp/entry/2025/01/12/222618>

## 作業手順計画

1. 表面の高難度部品
    1. J1 Type-C Receptacle
        * テスターでテスト
    1. U1 MCU Raspberry Pi RP2350A
        * 裏面のGNDのハンダ付け&テスト
    1. Y1 Crystal 12MHz
1. 表面の標準難度部品(部品高が低いものを優先)
    1. R1    Resistor 200Ω
    1. R2    Resistor 1Ω
    1. R3-4  Resistor 27Ω
    1. R5    Resistor 33Ω
    1. R6    Resistor 1kΩ
    1. C1    Capacitor 0.1uF
    1. C2-4  Capacitor 4.7uF
    1. C5-6  Capacitor 7-33pF
    1. L1    Inductor 3.3uH
    1. D1    LED Blue
    1. SW1-2 Button SKRPABE010
1.  裏面の高難度部品
    1. U4   Flash W25Q32JVUU
1. 裏面の標準難度部品(部品高が低いものを優先)
    1. R10       Resistor 1kΩ
    1. R7-8      Resistor 5.1kΩ
    1. R9,R11    Resistor 10kΩ
    1. C18       Capacitor 4.7uF
    1. C7-8      Capacitor 10uF
    1. C9-17,C19 Capacitor 0.1uF
    1. U2        USB Power Protection IC CH213K
    1. U3        Regulator 3.3V SOT-89 AMS1117-3.3
    1. J2        Box Pin Header 2x5 Pitch 1.27mm

## 作業記録動画

* その1: USBレセプタクル、RP2350A、クリスタルのハンダ付け <https://www.youtube.com/watch?v=hy3x_fPZgcA>
* その2: 残りの部品のハンダ付け <https://www.youtube.com/watch?v=1Ca5SVNlbrk>

PCに接続して認識されるところまで確認した。
(この時点ではプログラムが書き込めるかはわからない)

その後 [pico-examplesのblink\_simple](https://github.com/raspberrypi/pico-examples/tree/master/blink_simple) が動くことを確認した。
