# JavaでのSIMD利用について

JDK 21 の時点で2つの方法がある。

* HotSpot JavaVMの最適化でSIMDを使う
* jdk.incubator.vector を使う
* JavaからGPU

## TL;DR

* SIMDを使うならば [`jdk.incubator.vector`](https://docs.oracle.com/javase/jp/21/docs/api/jdk.incubator.vector/jdk/incubator/vector/package-summary.html) パッケージを使って書くのが良い。
    * JVMのJIT最適化に頼るとSIMDが有効活用されないケースが多い
* GPUを使う手法はあるがJNIを使ったモジュールが必須


## HotSpot JavaVMの最適化でSIMDを使う

詳細は [記事: HotSpot JavaVM の SIMD 最適化を効かせる方法](https://qiita.com/torao@github/items/be883ca5486a41fe96d6)

記事中のサンプルをOpenJDK 21.0.2を用いて手元で実行してみた。
結果floatでは約2.45倍、byteでは約9.48倍、SIMDにより高速化された。

<details>
<summary>float用のソースコードと実行結果</summary>

floatを対象としたコード:

```java
import java.util.Arrays;

public class J8SIMD {
    private static final int SIZE = 1024 * 1024;
    private static final float[] a = new float[SIZE];
    private static final float[] b = new float[SIZE];
    static {
        Arrays.fill(a, (float)1);
        Arrays.fill(b, (float)2);
    }
    public static void vectorAdd(){
        for(int i=0; i<a.length; i++){
            a[i] += b[i];
        }
    }
    public static void main(String[] args){
        // warming up
        for(int i=0; i<100; i++) vectorAdd();
        // measure
        long t0 = System.currentTimeMillis();
        for(int i=0; i<10000; i++){
            vectorAdd();
        }
        long t1 = System.currentTimeMillis();
        System.out.printf("vectorAdd: %,d[msec]", t1 - t0);
    }
}
```

floatを対象とした実行結果:

```console
$ javac J8SIMD.java

$ java -XX:+UseSuperWord J8SIMD
vectorAdd: 1,529[msec]

$ java -XX:-UseSuperWord J8SIMD
vectorAdd: 3,745[msec]
```

SIMDを使うことで約2.45倍高速化している。
</details>


<details>
<summary>byte用のソースコードと実行結果</summary>

byteを対象としたコード:

```java
import java.util.Arrays;

public class J8SIMD {
    private static final int SIZE = 1024 * 1024;
    private static final byte[] a = new byte[SIZE];
    private static final byte[] b = new byte[SIZE];
    static {
        Arrays.fill(a, (byte)1);
        Arrays.fill(b, (byte)2);
    }
    public static void vectorAdd(){
        for(int i=0; i<a.length; i++){
            a[i] += b[i];
        }
    }
    public static void main(String[] args){
        // warming up
        for(int i=0; i<100; i++) vectorAdd();
        // measure
        long t0 = System.currentTimeMillis();
        for(int i=0; i<10000; i++){
            vectorAdd();
        }
        long t1 = System.currentTimeMillis();
        System.out.printf("vectorAdd: %,d[msec]", t1 - t0);
    }
}
```

byteを対象とした実行結果:

```console
$ javac J8SIMD.java

$ java -XX:+UseSuperWord J8SIMD
vectorAdd: 360[msec]

$ java -XX:-UseSuperWord J8SIMD
vectorAdd: 3,411[msec]
```

SIMDを使うことで約9.48倍高速化している。
</details>

SIMDによる最適化が適用される条件は大まかに以下の通り:

* ループの全パラメータ(初期値、最終値、増減値)が固定であるもの
* ループ内に関数呼び出しがないこと
* 操作対象が配列であること
* インデックスが連続していること
* 単純な四則演算であること
* (Java 9 以降は)一部 aggregation にも対応しているらしい

その他

* デフォルトで有効
* `-XX:-UseSuperWord` で無効化

### 考察

* byte(8ビットデータ)にすると約10倍というのは想像以上
* 128ビットレジスタを使っているので…
    * floatは理論値で128/32で4倍 (実倍率 2.4)
    * byteは理論値で128/8で16倍 (実倍率 9.5)
    * オーバーヘッドを考えれば実倍率は極めて妥当
* CUDAを使う方法、あったりしない?
    * 数値計算専門ライブラリも調べたほうが良さそう

## jdk.incubator.vector を使う

参考: [JavaDoc jdk.incubator.vector](https://docs.oracle.com/javase/jp/21/docs/api/jdk.incubator.vector/jdk/incubator/vector/package-summary.html)

とりあえず前述のコードを jdk.incubator.vector を用いて書き直して計測した結果、おおよそ同等の速度が出た。

<details>
<summary>floatのSIMD演算</summary>

```java
import java.util.Arrays;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;

public class SIMDVectorFloat {
    static final int SIZE = 1024 * 1024;
    static final float[] a = new float[SIZE];
    static final float[] b = new float[SIZE];

    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    static {
        Arrays.fill(a, (float)1);
        Arrays.fill(b, (float)2);
    }

    public static void vectorAdd(){
        for (int i = 0; i < a.length; i += SPECIES.length()) {
            VectorMask<Float> m = SPECIES.indexInRange(i, a.length);
            FloatVector va = FloatVector.fromArray(SPECIES, a, i, m);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i, m);
            FloatVector vc = va.add(vb);
            vc.intoArray(a, i);
        }
    }
    public static void main(String[] args){
        // warming up
        for(int i=0; i<100; i++) vectorAdd();
        // measure
        long t0 = System.currentTimeMillis();
        for(int i=0; i<10000; i++){
            vectorAdd();
        }
        long t1 = System.currentTimeMillis();
        System.out.printf("vectorAdd: %,d[msec]", t1 - t0);
    }
}
```

```console
$ javac --add-modules jdk.incubator.vector SIMDVectorFloat.java
警告: 実験的なモジュールを使用しています: jdk.incubator.vector
警告1個

$ java  --add-modules jdk.incubator.vector SIMDVectorFloat
WARNING: Using incubator modules: jdk.incubator.vector
vectorAdd: 1,759[msec]
```
</details>

<details>
<summary>byteのSIMD演算</summary>

```java
import java.util.Arrays;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;

public class SIMDVectorByte {
    static final int SIZE = 1024 * 1024;
    static final byte[] a = new byte[SIZE];
    static final byte[] b = new byte[SIZE];

    static final VectorSpecies<Byte> SPECIES = ByteVector.SPECIES_PREFERRED;

    static {
        Arrays.fill(a, (byte)1);
        Arrays.fill(b, (byte)2);
    }

    public static void vectorAdd(){
        for (int i = 0; i < a.length; i += SPECIES.length()) {
            VectorMask<Byte> m = SPECIES.indexInRange(i, a.length);
            ByteVector va = ByteVector.fromArray(SPECIES, a, i, m);
            ByteVector vb = ByteVector.fromArray(SPECIES, b, i, m);
            ByteVector vc = va.add(vb);
            vc.intoArray(a, i);
        }
    }
    public static void main(String[] args){
        // warming up
        for(int i=0; i<100; i++) vectorAdd();
        // measure
        long t0 = System.currentTimeMillis();
        for(int i=0; i<10000; i++){
            vectorAdd();
        }
        long t1 = System.currentTimeMillis();
        System.out.printf("vectorAdd: %,d[msec]", t1 - t0);
    }
}
```

```console
$ javac --add-modules jdk.incubator.vector SIMDVectorByte.java
警告: 実験的なモジュールを使用しています: jdk.incubator.vector
警告1個

$ java --add-modules jdk.incubator.vector SIMDVectorByte.java
WARNING: Using incubator modules: jdk.incubator.vector
vectorAdd: 347[msec]
```
</details>

### 考察

* オーバーヘッドはほぼ同等か、じゃっかん大きいかも
* 明示的にSIMDが使えるのは良い
* 利用可能な演算が明確なのも良い
* コードが一見なにしてるのかわからないのは良くない
* C/C++のSIMD intrinsicsに比べて抽象度が高い
* incubatorなため将来性に不安がある

## JavaでGPU (CUDA)を使う方法の調査

簡単にサーベイを。

* [JavaによるGPUプログラミング 2020-04-24](https://blogs.oracle.com/otnjp/post/programming-the-gpu-in-java-ja)
* [Java で CUDA を導入する手順メモ 2015-06-21](https://kano.arkoak.com/2015/06/21/jcuda/)
* [jcuda.org](http://javagl.de/jcuda.org/)
* [Java bindings for OpenCL](http://www.jocl.org/)
* [TornadoVM](https://www.tornadovm.org/)

## サンプル

[koron/JavaSIMDTest](https://github.com/koron/JavaSIMDTest)

やってること

* SIMD最適化が効いてるかどうかの、2013年の検証 & その更新
* 距離関数をSIMDで書いてみてそのパフォーマンス評価

わかったこと

* 集約(aggregation)処理があるとSIMD最適化が効きにくい
* 自前で jdk.incubator.vector を使えば高速に書ける
    * ただしcosine距離はいまひとつ速くならない
    * 変数(XMMレジスタ)か集約処理が多すぎる?

## コンパイラ・コントロール

`-XX:+UnlockDiagnosticVMOptions` を指定すると[コンパイラ・コントロール](https://docs.oracle.com/javase/jp/21/vm/compiler-control1.html)が有効になる。

`-XX:+CompilerDirectivesPrint` を指定するとコンパイラ・コントロールのディレクティブが確認できる。
その中に `Vectorize` というのがあり、何かしらのコントロールが効きそう。

わかったこと

* 浮動小数点数の計算にAVX(SIMD)命令を利用している
* AVXがもつ大量のレジスタを全て有効活用しようとする
    * 直列に全部読み込んでレジスタ上でなんとかしようとしてる
* 並列演算に利用するレジスタが足りず、並列演算できない

### 再現手法と解説

目的: ベクトル間の距離計算がどのようにSIMD化されたのか、またはされてないかを観測する

1. [koron/JavaSIMDTest](https://github.com/koron/JavaSIMDTest) をチェックアウト

    * `DistanceBenchmark.Algorithm` クラスの `(simd|normal)(L2|Cos|DP)` メソッドで、ベクトル間距離の計算の3つアルゴリズムを、SIMDの有り無し、計6つのバリエーションとして実装している

2. OpenJDK 21を利用
3. hsdis をインストール (参考: https://chriswhocodes.com/hsdis/): ディスアセンブルに必要。これがないと16進数を読まされる
4. コンパイル `make DistanceBenchmark.class`
5. (JIT) コンパイラ・コントロール ディレクティブファイル directive.json を以下の内容で用意する

    ```json
    [
      {
        "match": [
          "*Algorithm.normal*",
          "*Algorithm.simd*"
        ],
        "c2": {
          "PrintAssembly": true,
          "PrintNMethods": true,
          "Vectorize": true
        }
      }
    ]
    ```

    これは特定のメソッドのみ、アセンブル言語リストを出力するというもの

6. プログラムを実行し、JITコンパイルされたアセンブル言語リストを確認する

    ```console
    $ java \
        -XX:+UnlockDiagnosticVMOptions \
        -XX:CompilerDirectivesFile=directive.json \
        -XX:+CompilerDirectivesPrint \
        --add-modules jdk.incubator.vector \
        DistanceBenchmark > out.log
    ```

    オプションの意味:

    * `-XX:+UnlockDiagnosticVMOptions` コンパイラ・コントロール・ディレクティブを使えるようにする
    * `-XX:CompilerDirectivesFile=directive.json` ディレクティブファイルを指定する
    * `-XX:+CompilerDirectivesPrint` 実際に適用されるディレクティブの情報を表示する

    out.log に諸々の出力がなされる。
    手元で実行した[サンプル](./out.log)を添付しておく。

#### out.logからの抜粋

JVMにはc1とc2の2種類のJITコンパイルがある。c1は実行前の粗いコンパイル。c2はある程度実行したのちの、実行プロファイルに基づいたより最適化するコンパイル。さらにc2は複数回行われる。以下では主に2度目のc2コンパイルを取り上げる。

以下は normalL2 からの抜粋。 
`vmovss` でメモリからAVXのレジスタに読み込み、
`vsubss` で引き算し `vmulss` で掛け算して `vaddss`
で足しこんでいることが見て取れる。
途中の計算結果を多数のAVXレジスタに保持し続け、速度を稼ごうとしているのがわかる。
また多数のレジスタを使ってしまうため、自動で並列化する猶予がないように見える。

```
  0x0000020ce0117434:   vmovss 0x24(%rdx,%r11,4),%xmm1
  0x0000020ce011743b:   vsubss 0x24(%r8,%r11,4),%xmm1,%xmm2
  0x0000020ce0117442:   vmovss 0x14(%rdx,%r11,4),%xmm1
  0x0000020ce0117449:   vsubss 0x14(%r8,%r11,4),%xmm1,%xmm7
  0x0000020ce0117450:   vmovss 0x10(%rdx,%r11,4),%xmm3
  0x0000020ce0117457:   vsubss 0x10(%r8,%r11,4),%xmm3,%xmm8
  0x0000020ce011745e:   vmovss 0x18(%rdx,%r11,4),%xmm1
  0x0000020ce0117465:   vsubss 0x18(%r8,%r11,4),%xmm1,%xmm9
  0x0000020ce011746c:   vmovss 0x1c(%rdx,%r11,4),%xmm3
  0x0000020ce0117473:   vsubss 0x1c(%r8,%r11,4),%xmm3,%xmm1
  0x0000020ce011747a:   vmovss 0x20(%rdx,%r11,4),%xmm4
  0x0000020ce0117481:   vsubss 0x20(%r8,%r11,4),%xmm4,%xmm3
  0x0000020ce0117488:   vmovss 0x28(%rdx,%r11,4),%xmm5
  0x0000020ce011748f:   vsubss 0x28(%r8,%r11,4),%xmm5,%xmm4
  0x0000020ce0117496:   vmovss 0x2c(%rdx,%r11,4),%xmm6
  0x0000020ce011749d:   vsubss 0x2c(%r8,%r11,4),%xmm6,%xmm5 ;*fsub {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::normalL2@17 (line 33)
  0x0000020ce01174a4:   vmulss %xmm2,%xmm2,%xmm10
  0x0000020ce01174a8:   vmulss %xmm5,%xmm5,%xmm5
  0x0000020ce01174ac:   vmulss %xmm4,%xmm4,%xmm11
  0x0000020ce01174b0:   vmulss %xmm3,%xmm3,%xmm6
  0x0000020ce01174b4:   vmulss %xmm1,%xmm1,%xmm2
  0x0000020ce01174b8:   vmulss %xmm9,%xmm9,%xmm1
  0x0000020ce01174bd:   vmulss %xmm8,%xmm8,%xmm4
  0x0000020ce01174c2:   vmulss %xmm7,%xmm7,%xmm3
  0x0000020ce01174c6:   vaddss %xmm0,%xmm4,%xmm4
  0x0000020ce01174ca:   vaddss %xmm3,%xmm4,%xmm0
  0x0000020ce01174ce:   vaddss %xmm0,%xmm1,%xmm1
  0x0000020ce01174d2:   vaddss %xmm1,%xmm2,%xmm1
  0x0000020ce01174d6:   vaddss %xmm1,%xmm6,%xmm0
  0x0000020ce01174da:   vaddss %xmm10,%xmm0,%xmm1
  0x0000020ce01174df:   vaddss %xmm1,%xmm11,%xmm0
  0x0000020ce01174e3:   vaddss %xmm0,%xmm5,%xmm0            ;*fadd {reexecute=0 rethrow=0 return_oop=0}
```

<details>
<summary>参考: 2回目のc2 normalL2 ディスアセンブリ全文</summary>

全体を見るとアライメントを考慮し、アライメント前の部分、
アライメントされている部分、されていない部分と個別に計算しているように見える。
これは近年良くある最適化の1つであると考えて良さそう。

```
[Disassembly]
--------------------------------------------------------------------------------

[Constant Pool]
             Address          hex4                    hex8      
  0x0000020ce01173a0:   0x00000000      0xf4f4f4f400000000      
  0x0000020ce01173a4:   0xf4f4f4f4                              
  0x0000020ce01173a8:   0xf4f4f4f4      0xf4f4f4f4f4f4f4f4      
  0x0000020ce01173ac:   0xf4f4f4f4                              
  0x0000020ce01173b0:   0xf4f4f4f4      0xf4f4f4f4f4f4f4f4      
  0x0000020ce01173b4:   0xf4f4f4f4                              
  0x0000020ce01173b8:   0xf4f4f4f4      0xf4f4f4f4f4f4f4f4      
  0x0000020ce01173bc:   0xf4f4f4f4                              

--------------------------------------------------------------------------------

[Verified Entry Point]
  # {method} {0x0000020cc1402a40} 'normalL2' '([F[F)F' in 'DistanceBenchmark$Algorithm'
  # parm0:    rdx:rdx   = '[F'
  # parm1:    r8:r8     = '[F'
  #           [sp+0x30]  (sp of caller)
  0x0000020ce01173c0:   mov    %eax,-0x8000(%rsp)           ;   {no_reloc}
  0x0000020ce01173c7:   push   %rbp
  0x0000020ce01173c8:   sub    $0x20,%rsp
  0x0000020ce01173cc:   cmpl   $0x1,0x20(%r15)
  0x0000020ce01173d4:   jne    0x0000020ce011756e           ;*synchronization entry
                                                            ; - DistanceBenchmark$Algorithm::normalL2@-1 (line 31)
  0x0000020ce01173da:   mov    0xc(%rdx),%r11d              ; implicit exception: dispatches to 0x0000020ce011753f
                                                            ;*faload {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::normalL2@13 (line 33)
  0x0000020ce01173de:   test   %r11d,%r11d
  0x0000020ce01173e1:   jbe    0x0000020ce011753f
  0x0000020ce01173e7:   cmp    $0x7cf,%r11d
  0x0000020ce01173ee:   jbe    0x0000020ce011753f
  0x0000020ce01173f4:   mov    0xc(%r8),%r10d               ; implicit exception: dispatches to 0x0000020ce011753f
  0x0000020ce01173f8:   test   %r10d,%r10d
  0x0000020ce01173fb:   jbe    0x0000020ce011753f
  0x0000020ce0117401:   cmp    $0x7cf,%r10d
  0x0000020ce0117408:   jbe    0x0000020ce011753f
  0x0000020ce011740e:   vmovss 0x10(%r8),%xmm1              ;*faload {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::normalL2@16 (line 33)
  0x0000020ce0117414:   vmovss 0x10(%rdx),%xmm0
  0x0000020ce0117419:   vsubss %xmm1,%xmm0,%xmm1            ;*fsub {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::normalL2@17 (line 33)
  0x0000020ce011741d:   vmulss %xmm1,%xmm1,%xmm0
  0x0000020ce0117421:   mov    $0x1,%r11d
  0x0000020ce0117427:   vaddss -0x8f(%rip),%xmm0,%xmm0        # 0x0000020ce01173a0
                                                            ;*fadd {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::normalL2@26 (line 34)
                                                            ;   {section_word}
  0x0000020ce011742f:   jmp    0x0000020ce0117434
  0x0000020ce0117431:   mov    %r9d,%r11d                   ;*aload_0 {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::normalL2@11 (line 33)
  0x0000020ce0117434:   vmovss 0x24(%rdx,%r11,4),%xmm1
  0x0000020ce011743b:   vsubss 0x24(%r8,%r11,4),%xmm1,%xmm2
  0x0000020ce0117442:   vmovss 0x14(%rdx,%r11,4),%xmm1
  0x0000020ce0117449:   vsubss 0x14(%r8,%r11,4),%xmm1,%xmm7
  0x0000020ce0117450:   vmovss 0x10(%rdx,%r11,4),%xmm3
  0x0000020ce0117457:   vsubss 0x10(%r8,%r11,4),%xmm3,%xmm8
  0x0000020ce011745e:   vmovss 0x18(%rdx,%r11,4),%xmm1
  0x0000020ce0117465:   vsubss 0x18(%r8,%r11,4),%xmm1,%xmm9
  0x0000020ce011746c:   vmovss 0x1c(%rdx,%r11,4),%xmm3
  0x0000020ce0117473:   vsubss 0x1c(%r8,%r11,4),%xmm3,%xmm1
  0x0000020ce011747a:   vmovss 0x20(%rdx,%r11,4),%xmm4
  0x0000020ce0117481:   vsubss 0x20(%r8,%r11,4),%xmm4,%xmm3
  0x0000020ce0117488:   vmovss 0x28(%rdx,%r11,4),%xmm5
  0x0000020ce011748f:   vsubss 0x28(%r8,%r11,4),%xmm5,%xmm4
  0x0000020ce0117496:   vmovss 0x2c(%rdx,%r11,4),%xmm6
  0x0000020ce011749d:   vsubss 0x2c(%r8,%r11,4),%xmm6,%xmm5 ;*fsub {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::normalL2@17 (line 33)
  0x0000020ce01174a4:   vmulss %xmm2,%xmm2,%xmm10
  0x0000020ce01174a8:   vmulss %xmm5,%xmm5,%xmm5
  0x0000020ce01174ac:   vmulss %xmm4,%xmm4,%xmm11
  0x0000020ce01174b0:   vmulss %xmm3,%xmm3,%xmm6
  0x0000020ce01174b4:   vmulss %xmm1,%xmm1,%xmm2
  0x0000020ce01174b8:   vmulss %xmm9,%xmm9,%xmm1
  0x0000020ce01174bd:   vmulss %xmm8,%xmm8,%xmm4
  0x0000020ce01174c2:   vmulss %xmm7,%xmm7,%xmm3
  0x0000020ce01174c6:   vaddss %xmm0,%xmm4,%xmm4
  0x0000020ce01174ca:   vaddss %xmm3,%xmm4,%xmm0
  0x0000020ce01174ce:   vaddss %xmm0,%xmm1,%xmm1
  0x0000020ce01174d2:   vaddss %xmm1,%xmm2,%xmm1
  0x0000020ce01174d6:   vaddss %xmm1,%xmm6,%xmm0
  0x0000020ce01174da:   vaddss %xmm10,%xmm0,%xmm1
  0x0000020ce01174df:   vaddss %xmm1,%xmm11,%xmm0
  0x0000020ce01174e3:   vaddss %xmm0,%xmm5,%xmm0            ;*fadd {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::normalL2@26 (line 34)
  0x0000020ce01174e7:   lea    0x8(%r11),%r9d
  0x0000020ce01174eb:   cmp    $0x7c9,%r9d
  0x0000020ce01174f2:   jl     0x0000020ce0117431           ;*goto {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::normalL2@31 (line 32)
  0x0000020ce01174f8:   mov    0x458(%r15),%r10             ; ImmutableOopMap {r8=Oop rdx=Oop }
                                                            ;*goto {reexecute=1 rethrow=0 return_oop=0}
                                                            ; - (reexecute) DistanceBenchmark$Algorithm::normalL2@31 (line 32)
  0x0000020ce01174ff:   test   %eax,(%r10)                  ;*goto {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::normalL2@31 (line 32)
                                                            ;   {poll}
  0x0000020ce0117502:   add    $0x8,%r11d                   ;*aload_0 {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::normalL2@11 (line 33)
  0x0000020ce0117506:   vmovss 0x10(%rdx,%r11,4),%xmm1
  0x0000020ce011750d:   vsubss 0x10(%r8,%r11,4),%xmm1,%xmm2 ;*fsub {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::normalL2@17 (line 33)
  0x0000020ce0117514:   vmulss %xmm2,%xmm2,%xmm1
  0x0000020ce0117518:   vaddss %xmm1,%xmm0,%xmm0            ;*fadd {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::normalL2@26 (line 34)
  0x0000020ce011751c:   inc    %r11d                        ;*iinc {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::normalL2@28 (line 32)
  0x0000020ce011751f:   cmp    $0x7d0,%r11d
  0x0000020ce0117526:   jl     0x0000020ce0117506           ;*if_icmpge {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::normalL2@8 (line 32)
  0x0000020ce0117528:   vsqrtss %xmm0,%xmm0,%xmm0           ;*d2f {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::normalL2@39 (line 36)
  0x0000020ce011752c:   add    $0x20,%rsp
  0x0000020ce0117530:   pop    %rbp
  0x0000020ce0117531:   cmp    0x450(%r15),%rsp             ;   {poll_return}
  0x0000020ce0117538:   ja     0x0000020ce0117558
  0x0000020ce011753e:   ret                                 ;*if_icmpge {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::normalL2@8 (line 32)
  0x0000020ce011753f:   mov    %rdx,%rbp
  0x0000020ce0117542:   mov    %r8,(%rsp)
  0x0000020ce0117546:   mov    $0xffffff76,%edx
  0x0000020ce011754b:   call   0x0000020cdfa04600           ; ImmutableOopMap {rbp=Oop [0]=Oop }
                                                            ;*if_icmpge {reexecute=1 rethrow=0 return_oop=0}
                                                            ; - (reexecute) DistanceBenchmark$Algorithm::normalL2@8 (line 32)
                                                            ;   {runtime_call UncommonTrapBlob}
  0x0000020ce0117550:   nopl   0x1000340(%rax,%rax,1)       ;*if_icmpge {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::normalL2@8 (line 32)
                                                            ;   {other}
  0x0000020ce0117558:   movabs $0x20ce0117531,%r10          ;   {internal_word}
  0x0000020ce0117562:   mov    %r10,0x468(%r15)
  0x0000020ce0117569:   jmp    0x0000020cdfa05700           ;   {runtime_call SafepointBlob}
  0x0000020ce011756e:   call   Stub::nmethod_entry_barrier  ;   {runtime_call StubRoutines (final stubs)}
  0x0000020ce0117573:   jmp    0x0000020ce01173da
[Exception Handler]
  0x0000020ce0117578:   jmp    0x0000020cdfacdd00           ;   {no_reloc}
[Deopt Handler Code]
  0x0000020ce011757d:   call   0x0000020ce0117582
  0x0000020ce0117582:   subq   $0x5,(%rsp)
  0x0000020ce0117587:   jmp    0x0000020cdfa049a0           ;   {runtime_call DeoptimizationBlob}
  0x0000020ce011758c:   hlt    
  0x0000020ce011758d:   hlt    
  0x0000020ce011758e:   hlt    
  0x0000020ce011758f:   hlt    
--------------------------------------------------------------------------------
[/Disassembly]
```
</details>

次に simd を用いた simdL2 のディスアセンブリの抜粋を見てみる。
`vsubps` や `vmulps` など命令のサフィックスが `ps` に変わったこと、
扱ってるレジスタ名が `xmm{n}` から `ymm{n}` に変わったことから、
コードの意図通りにSIMD計算が用いられていることがわかる。

参考: <https://www.torutk.com/projects/swe/wiki/X86CPU#AVX>

```
  0x0000020ce0140697:   vmaskmovps 0x10(%rdx),%ymm5,%ymm1   ;*invokestatic loadMasked {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::fromArray0Template@52 (line 3303)
                                                            ; - jdk.incubator.vector.Float256Vector::fromArray0@11 (line 856)
                                                            ; - jdk.incubator.vector.FloatVector::fromArray@32 (line 2830)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@32 (line 44)
  0x0000020ce014069d:   movslq %r8d,%r8
  0x0000020ce01406a0:   add    $0xfffffffffffffff8,%r8
  0x0000020ce01406a4:   test   %r8,%r8
  0x0000020ce01406a7:   jl     0x0000020ce0140a80
  0x0000020ce01406ad:   vmaskmovps 0x10(%r9),%ymm5,%ymm0    ;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce01406b3:   vsubps %ymm0,%ymm1,%ymm0            ;*invokestatic binaryOp {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::lanewiseTemplate@96 (line 774)
                                                            ; - jdk.incubator.vector.Float256Vector::lanewise@3 (line 285)
                                                            ; - jdk.incubator.vector.Float256Vector::lanewise@3 (line 41)
                                                            ; - jdk.incubator.vector.FloatVector::sub@5 (line 1298)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@53 (line 46)
  0x0000020ce01406b7:   vmulps %ymm0,%ymm0,%ymm0
  0x0000020ce01406bb:   mov    $0x8,%ecx
```

<details>
<summary>参考: 2回目のc2 normalL2 ディスアセンブリ全文</summary>

```
[Disassembly]
--------------------------------------------------------------------------------

[Constant Pool]
             Address          hex4                    hex8      
  0x0000020ce0140620:   0x00000000      0xf4f4f4f400000000      
  0x0000020ce0140624:   0xf4f4f4f4                              
  0x0000020ce0140628:   0xf4f4f4f4      0xf4f4f4f4f4f4f4f4      
  0x0000020ce014062c:   0xf4f4f4f4                              
  0x0000020ce0140630:   0xf4f4f4f4      0xf4f4f4f4f4f4f4f4      
  0x0000020ce0140634:   0xf4f4f4f4                              
  0x0000020ce0140638:   0xf4f4f4f4      0xf4f4f4f4f4f4f4f4      
  0x0000020ce014063c:   0xf4f4f4f4                              

--------------------------------------------------------------------------------

[Verified Entry Point]
  # {method} {0x0000020cc1402b58} 'simdL2' '([F[F)F' in 'DistanceBenchmark$Algorithm'
  # parm0:    rdx:rdx   = '[F'
  # parm1:    r8:r8     = '[F'
  #           [sp+0x70]  (sp of caller)
  0x0000020ce0140640:   mov    %eax,-0x8000(%rsp)           ;   {no_reloc}
  0x0000020ce0140647:   push   %rbp
  0x0000020ce0140648:   sub    $0x60,%rsp
  0x0000020ce014064c:   cmpl   $0x1,0x20(%r15)
  0x0000020ce0140654:   jne    0x0000020ce0140abe           ;*synchronization entry
                                                            ; - DistanceBenchmark$Algorithm::simdL2@-1 (line 41)
  0x0000020ce014065a:   mov    %rdx,%r11
  0x0000020ce014065d:   mov    0xc(%rdx),%r10d              ; implicit exception: dispatches to 0x0000020ce0140a86
  0x0000020ce0140661:   mov    %r8,%r9
  0x0000020ce0140664:   mov    0xc(%r8),%r8d                ; implicit exception: dispatches to 0x0000020ce0140a89
  0x0000020ce0140668:   movslq %r10d,%r10
  0x0000020ce014066b:   add    $0xfffffffffffffff8,%r10     ;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce014066f:   vpcmpeqd %ymm5,%ymm5,%ymm5          ;*invokestatic fromBitsCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.Float256Vector$Float256Mask::maskAll@25 (line 765)
                                                            ; - jdk.incubator.vector.FloatVector$FloatSpecies::maskAll@71 (line 3899)
                                                            ; - jdk.incubator.vector.AbstractSpecies::indexInRange@2 (line 216)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@18 (line 43)
  0x0000020ce0140673:   xor    %ebx,%ebx
  0x0000020ce0140675:   xor    %ecx,%ecx
  0x0000020ce0140677:   vpxor  %xmm0,%xmm0,%xmm0
  0x0000020ce014067b:   vextracti128 $0x1,%ymm5,%xmm3
  0x0000020ce0140681:   vpackssdw %xmm3,%xmm5,%xmm3
  0x0000020ce0140685:   vpacksswb %xmm0,%xmm3,%xmm3
  0x0000020ce0140689:   vpabsb %xmm3,%xmm3
  0x0000020ce014068e:   test   %r10,%r10                    ;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce0140691:   jl     0x0000020ce0140a77
  0x0000020ce0140697:   vmaskmovps 0x10(%rdx),%ymm5,%ymm1   ;*invokestatic loadMasked {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::fromArray0Template@52 (line 3303)
                                                            ; - jdk.incubator.vector.Float256Vector::fromArray0@11 (line 856)
                                                            ; - jdk.incubator.vector.FloatVector::fromArray@32 (line 2830)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@32 (line 44)
  0x0000020ce014069d:   movslq %r8d,%r8
  0x0000020ce01406a0:   add    $0xfffffffffffffff8,%r8
  0x0000020ce01406a4:   test   %r8,%r8
  0x0000020ce01406a7:   jl     0x0000020ce0140a80
  0x0000020ce01406ad:   vmaskmovps 0x10(%r9),%ymm5,%ymm0    ;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce01406b3:   vsubps %ymm0,%ymm1,%ymm0            ;*invokestatic binaryOp {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::lanewiseTemplate@96 (line 774)
                                                            ; - jdk.incubator.vector.Float256Vector::lanewise@3 (line 285)
                                                            ; - jdk.incubator.vector.Float256Vector::lanewise@3 (line 41)
                                                            ; - jdk.incubator.vector.FloatVector::sub@5 (line 1298)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@53 (line 46)
  0x0000020ce01406b7:   vmulps %ymm0,%ymm0,%ymm0
  0x0000020ce01406bb:   mov    $0x8,%ecx
  0x0000020ce01406c0:   vxorps %xmm1,%xmm1,%xmm1
  0x0000020ce01406c4:   vaddss %xmm0,%xmm1,%xmm1
  0x0000020ce01406c8:   vpshufd $0x1,%xmm0,%xmm4
  0x0000020ce01406cd:   vaddss %xmm4,%xmm1,%xmm1
  0x0000020ce01406d1:   vpshufd $0x2,%xmm0,%xmm4
  0x0000020ce01406d6:   vaddss %xmm4,%xmm1,%xmm1
  0x0000020ce01406da:   vpshufd $0x3,%xmm0,%xmm4
  0x0000020ce01406df:   vaddss %xmm4,%xmm1,%xmm1
  0x0000020ce01406e3:   vextractf128 $0x1,%ymm0,%xmm4
  0x0000020ce01406e9:   vaddss %xmm4,%xmm1,%xmm1
  0x0000020ce01406ed:   vpshufd $0x1,%xmm4,%xmm2
  0x0000020ce01406f2:   vaddss %xmm2,%xmm1,%xmm1
  0x0000020ce01406f6:   vpshufd $0x2,%xmm4,%xmm2
  0x0000020ce01406fb:   vaddss %xmm2,%xmm1,%xmm1
  0x0000020ce01406ff:   vpshufd $0x3,%xmm4,%xmm2
  0x0000020ce0140704:   vaddss %xmm2,%xmm1,%xmm1            ;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce0140708:   vaddss -0xf0(%rip),%xmm1,%xmm0        # 0x0000020ce0140620
                                                            ;*fadd {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::simdL2@72 (line 47)
                                                            ;   {section_word}
  0x0000020ce0140710:   jmp    0x0000020ce0140714
  0x0000020ce0140712:   mov    %edi,%ecx                    ;*getstatic SPECIES {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::simdL2@11 (line 43)
  0x0000020ce0140714:   movslq %ecx,%rdi                    ;*i2l {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::fromArray@7 (line 2829)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@32 (line 44)
  0x0000020ce0140717:   cmp    %r10,%rdi
  0x0000020ce014071a:   jg     0x0000020ce01409ad           ;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce0140720:   vmaskmovps 0x10(%r11,%rdi,4),%ymm5,%ymm1;*invokestatic loadMasked {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::fromArray0Template@52 (line 3303)
                                                            ; - jdk.incubator.vector.Float256Vector::fromArray0@11 (line 856)
                                                            ; - jdk.incubator.vector.FloatVector::fromArray@32 (line 2830)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@32 (line 44)
  0x0000020ce0140727:   cmp    %r8,%rdi
  0x0000020ce014072a:   jg     0x0000020ce0140a04
  0x0000020ce0140730:   vmaskmovps 0x10(%r9,%rdi,4),%ymm5,%ymm2;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce0140737:   vsubps %ymm2,%ymm1,%ymm1            ;*invokestatic binaryOp {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::lanewiseTemplate@96 (line 774)
                                                            ; - jdk.incubator.vector.Float256Vector::lanewise@3 (line 285)
                                                            ; - jdk.incubator.vector.Float256Vector::lanewise@3 (line 41)
                                                            ; - jdk.incubator.vector.FloatVector::sub@5 (line 1298)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@53 (line 46)
  0x0000020ce014073b:   vmulps %ymm1,%ymm1,%ymm1
  0x0000020ce014073f:   lea    0x8(%rdi),%rbx
  0x0000020ce0140743:   vxorps %xmm2,%xmm2,%xmm2
  0x0000020ce0140747:   vaddss %xmm1,%xmm2,%xmm2
  0x0000020ce014074b:   vpshufd $0x1,%xmm1,%xmm6
  0x0000020ce0140750:   vaddss %xmm6,%xmm2,%xmm2
  0x0000020ce0140754:   vpshufd $0x2,%xmm1,%xmm6
  0x0000020ce0140759:   vaddss %xmm6,%xmm2,%xmm2
  0x0000020ce014075d:   vpshufd $0x3,%xmm1,%xmm6
  0x0000020ce0140762:   vaddss %xmm6,%xmm2,%xmm2
  0x0000020ce0140766:   vextractf128 $0x1,%ymm1,%xmm6
  0x0000020ce014076c:   vaddss %xmm6,%xmm2,%xmm2
  0x0000020ce0140770:   vpshufd $0x1,%xmm6,%xmm4
  0x0000020ce0140775:   vaddss %xmm4,%xmm2,%xmm2
  0x0000020ce0140779:   vpshufd $0x2,%xmm6,%xmm4
  0x0000020ce014077e:   vaddss %xmm4,%xmm2,%xmm2
  0x0000020ce0140782:   vpshufd $0x3,%xmm6,%xmm4
  0x0000020ce0140787:   vaddss %xmm4,%xmm2,%xmm2            ;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce014078b:   vaddss %xmm0,%xmm2,%xmm0            ;*fadd {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::simdL2@72 (line 47)
  0x0000020ce014078f:   cmp    %r10,%rbx
  0x0000020ce0140792:   jg     0x0000020ce01409a8           ;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce0140798:   vmaskmovps 0x30(%r11,%rdi,4),%ymm5,%ymm1;*invokestatic loadMasked {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::fromArray0Template@52 (line 3303)
                                                            ; - jdk.incubator.vector.Float256Vector::fromArray0@11 (line 856)
                                                            ; - jdk.incubator.vector.FloatVector::fromArray@32 (line 2830)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@32 (line 44)
  0x0000020ce014079f:   cmp    %r8,%rbx
  0x0000020ce01407a2:   jg     0x0000020ce01409ff
  0x0000020ce01407a8:   vmaskmovps 0x30(%r9,%rdi,4),%ymm5,%ymm2;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce01407af:   vsubps %ymm2,%ymm1,%ymm1            ;*invokestatic binaryOp {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::lanewiseTemplate@96 (line 774)
                                                            ; - jdk.incubator.vector.Float256Vector::lanewise@3 (line 285)
                                                            ; - jdk.incubator.vector.Float256Vector::lanewise@3 (line 41)
                                                            ; - jdk.incubator.vector.FloatVector::sub@5 (line 1298)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@53 (line 46)
  0x0000020ce01407b3:   vmulps %ymm1,%ymm1,%ymm2
  0x0000020ce01407b7:   lea    0x10(%rdi),%rbx
  0x0000020ce01407bb:   vxorps %xmm1,%xmm1,%xmm1
  0x0000020ce01407bf:   vaddss %xmm2,%xmm1,%xmm1
  0x0000020ce01407c3:   vpshufd $0x1,%xmm2,%xmm4
  0x0000020ce01407c8:   vaddss %xmm4,%xmm1,%xmm1
  0x0000020ce01407cc:   vpshufd $0x2,%xmm2,%xmm4
  0x0000020ce01407d1:   vaddss %xmm4,%xmm1,%xmm1
  0x0000020ce01407d5:   vpshufd $0x3,%xmm2,%xmm4
  0x0000020ce01407da:   vaddss %xmm4,%xmm1,%xmm1
  0x0000020ce01407de:   vextractf128 $0x1,%ymm2,%xmm4
  0x0000020ce01407e4:   vaddss %xmm4,%xmm1,%xmm1
  0x0000020ce01407e8:   vpshufd $0x1,%xmm4,%xmm6
  0x0000020ce01407ed:   vaddss %xmm6,%xmm1,%xmm1
  0x0000020ce01407f1:   vpshufd $0x2,%xmm4,%xmm6
  0x0000020ce01407f6:   vaddss %xmm6,%xmm1,%xmm1
  0x0000020ce01407fa:   vpshufd $0x3,%xmm4,%xmm6
  0x0000020ce01407ff:   vaddss %xmm6,%xmm1,%xmm1            ;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce0140803:   vaddss %xmm0,%xmm1,%xmm0            ;*fadd {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::simdL2@72 (line 47)
  0x0000020ce0140807:   cmp    %r10,%rbx                    ;   {no_reloc}
  0x0000020ce014080a:   jg     0x0000020ce01409b2           ;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce0140810:   vmaskmovps 0x50(%r11,%rdi,4),%ymm5,%ymm1;*invokestatic loadMasked {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::fromArray0Template@52 (line 3303)
                                                            ; - jdk.incubator.vector.Float256Vector::fromArray0@11 (line 856)
                                                            ; - jdk.incubator.vector.FloatVector::fromArray@32 (line 2830)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@32 (line 44)
  0x0000020ce0140817:   cmp    %r8,%rbx
  0x0000020ce014081a:   jg     0x0000020ce0140a09
  0x0000020ce0140820:   vmaskmovps 0x50(%r9,%rdi,4),%ymm5,%ymm2;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce0140827:   vsubps %ymm2,%ymm1,%ymm1            ;*invokestatic binaryOp {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::lanewiseTemplate@96 (line 774)
                                                            ; - jdk.incubator.vector.Float256Vector::lanewise@3 (line 285)
                                                            ; - jdk.incubator.vector.Float256Vector::lanewise@3 (line 41)
                                                            ; - jdk.incubator.vector.FloatVector::sub@5 (line 1298)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@53 (line 46)
  0x0000020ce014082b:   vmulps %ymm1,%ymm1,%ymm1
  0x0000020ce014082f:   lea    0x18(%rdi),%rbx
  0x0000020ce0140833:   vxorps %xmm2,%xmm2,%xmm2
  0x0000020ce0140837:   vaddss %xmm1,%xmm2,%xmm2
  0x0000020ce014083b:   vpshufd $0x1,%xmm1,%xmm4
  0x0000020ce0140840:   vaddss %xmm4,%xmm2,%xmm2
  0x0000020ce0140844:   vpshufd $0x2,%xmm1,%xmm4
  0x0000020ce0140849:   vaddss %xmm4,%xmm2,%xmm2
  0x0000020ce014084d:   vpshufd $0x3,%xmm1,%xmm4
  0x0000020ce0140852:   vaddss %xmm4,%xmm2,%xmm2
  0x0000020ce0140856:   vextractf128 $0x1,%ymm1,%xmm4
  0x0000020ce014085c:   vaddss %xmm4,%xmm2,%xmm2
  0x0000020ce0140860:   vpshufd $0x1,%xmm4,%xmm6
  0x0000020ce0140865:   vaddss %xmm6,%xmm2,%xmm2
  0x0000020ce0140869:   vpshufd $0x2,%xmm4,%xmm6
  0x0000020ce014086e:   vaddss %xmm6,%xmm2,%xmm2
  0x0000020ce0140872:   vpshufd $0x3,%xmm4,%xmm6
  0x0000020ce0140877:   vaddss %xmm6,%xmm2,%xmm2            ;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce014087b:   vaddss %xmm0,%xmm2,%xmm0            ;*fadd {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::simdL2@72 (line 47)
  0x0000020ce014087f:   cmp    %r10,%rbx
  0x0000020ce0140882:   jg     0x0000020ce01409a5           ;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce0140888:   vmaskmovps 0x70(%r11,%rdi,4),%ymm5,%ymm1;*invokestatic loadMasked {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::fromArray0Template@52 (line 3303)
                                                            ; - jdk.incubator.vector.Float256Vector::fromArray0@11 (line 856)
                                                            ; - jdk.incubator.vector.FloatVector::fromArray@32 (line 2830)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@32 (line 44)
  0x0000020ce014088f:   cmp    %r8,%rbx
  0x0000020ce0140892:   jg     0x0000020ce01409fc
  0x0000020ce0140898:   vmaskmovps 0x70(%r9,%rdi,4),%ymm5,%ymm2;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce014089f:   vsubps %ymm2,%ymm1,%ymm1            ;*invokestatic binaryOp {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::lanewiseTemplate@96 (line 774)
                                                            ; - jdk.incubator.vector.Float256Vector::lanewise@3 (line 285)
                                                            ; - jdk.incubator.vector.Float256Vector::lanewise@3 (line 41)
                                                            ; - jdk.incubator.vector.FloatVector::sub@5 (line 1298)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@53 (line 46)
  0x0000020ce01408a3:   vmulps %ymm1,%ymm1,%ymm2
  0x0000020ce01408a7:   vxorps %xmm1,%xmm1,%xmm1
  0x0000020ce01408ab:   vaddss %xmm2,%xmm1,%xmm1
  0x0000020ce01408af:   vpshufd $0x1,%xmm2,%xmm4
  0x0000020ce01408b4:   vaddss %xmm4,%xmm1,%xmm1
  0x0000020ce01408b8:   vpshufd $0x2,%xmm2,%xmm4
  0x0000020ce01408bd:   vaddss %xmm4,%xmm1,%xmm1
  0x0000020ce01408c1:   vpshufd $0x3,%xmm2,%xmm4
  0x0000020ce01408c6:   vaddss %xmm4,%xmm1,%xmm1
  0x0000020ce01408ca:   vextractf128 $0x1,%ymm2,%xmm4
  0x0000020ce01408d0:   vaddss %xmm4,%xmm1,%xmm1
  0x0000020ce01408d4:   vpshufd $0x1,%xmm4,%xmm6
  0x0000020ce01408d9:   vaddss %xmm6,%xmm1,%xmm1
  0x0000020ce01408dd:   vpshufd $0x2,%xmm4,%xmm6
  0x0000020ce01408e2:   vaddss %xmm6,%xmm1,%xmm1
  0x0000020ce01408e6:   vpshufd $0x3,%xmm4,%xmm6
  0x0000020ce01408eb:   vaddss %xmm6,%xmm1,%xmm1            ;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce01408ef:   vaddss %xmm0,%xmm1,%xmm0            ;*fadd {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::simdL2@72 (line 47)
  0x0000020ce01408f3:   lea    0x20(%rcx),%edi
  0x0000020ce01408f6:   cmp    $0x7b8,%edi
  0x0000020ce01408fc:   jl     0x0000020ce0140712
  0x0000020ce0140902:   add    $0x20,%ecx                   ;*getstatic SPECIES {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::simdL2@11 (line 43)
  0x0000020ce0140905:   movslq %ecx,%rbx                    ;*i2l {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::fromArray@7 (line 2829)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@32 (line 44)
                                                            ;   {no_reloc}
  0x0000020ce0140908:   cmp    %r10,%rbx
  0x0000020ce014090b:   jg     0x0000020ce0140a58           ;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce0140911:   vmaskmovps 0x10(%r11,%rbx,4),%ymm5,%ymm1;*invokestatic loadMasked {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::fromArray0Template@52 (line 3303)
                                                            ; - jdk.incubator.vector.Float256Vector::fromArray0@11 (line 856)
                                                            ; - jdk.incubator.vector.FloatVector::fromArray@32 (line 2830)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@32 (line 44)
  0x0000020ce0140918:   cmp    %r8,%rbx
  0x0000020ce014091b:   jg     0x0000020ce0140a66
  0x0000020ce0140921:   vmaskmovps 0x10(%r9,%rbx,4),%ymm5,%ymm2;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce0140928:   vsubps %ymm2,%ymm1,%ymm1            ;*invokestatic binaryOp {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::lanewiseTemplate@96 (line 774)
                                                            ; - jdk.incubator.vector.Float256Vector::lanewise@3 (line 285)
                                                            ; - jdk.incubator.vector.Float256Vector::lanewise@3 (line 41)
                                                            ; - jdk.incubator.vector.FloatVector::sub@5 (line 1298)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@53 (line 46)
  0x0000020ce014092c:   vmulps %ymm1,%ymm1,%ymm1
  0x0000020ce0140930:   vxorps %xmm2,%xmm2,%xmm2
  0x0000020ce0140934:   vaddss %xmm1,%xmm2,%xmm2
  0x0000020ce0140938:   vpshufd $0x1,%xmm1,%xmm4
  0x0000020ce014093d:   vaddss %xmm4,%xmm2,%xmm2
  0x0000020ce0140941:   vpshufd $0x2,%xmm1,%xmm4
  0x0000020ce0140946:   vaddss %xmm4,%xmm2,%xmm2
  0x0000020ce014094a:   vpshufd $0x3,%xmm1,%xmm4
  0x0000020ce014094f:   vaddss %xmm4,%xmm2,%xmm2
  0x0000020ce0140953:   vextractf128 $0x1,%ymm1,%xmm4
  0x0000020ce0140959:   vaddss %xmm4,%xmm2,%xmm2
  0x0000020ce014095d:   vpshufd $0x1,%xmm4,%xmm6
  0x0000020ce0140962:   vaddss %xmm6,%xmm2,%xmm2
  0x0000020ce0140966:   vpshufd $0x2,%xmm4,%xmm6
  0x0000020ce014096b:   vaddss %xmm6,%xmm2,%xmm2
  0x0000020ce014096f:   vpshufd $0x3,%xmm4,%xmm6
  0x0000020ce0140974:   vaddss %xmm6,%xmm2,%xmm2            ;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce0140978:   vaddss %xmm2,%xmm0,%xmm0            ;*fadd {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::simdL2@72 (line 47)
  0x0000020ce014097c:   add    $0x8,%ecx
  0x0000020ce014097f:   cmp    $0x7d0,%ecx
  0x0000020ce0140985:   jl     0x0000020ce0140905           ;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce014098b:   vsqrtss %xmm0,%xmm0,%xmm0           ;*d2f {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - DistanceBenchmark$Algorithm::simdL2@93 (line 49)
  0x0000020ce014098f:   vzeroupper 
  0x0000020ce0140992:   add    $0x60,%rsp
  0x0000020ce0140996:   pop    %rbp
  0x0000020ce0140997:   cmp    0x450(%r15),%rsp             ;   {poll_return}
  0x0000020ce014099e:   ja     0x0000020ce0140aa8
  0x0000020ce01409a4:   ret    
  0x0000020ce01409a5:   add    $0x10,%ecx
  0x0000020ce01409a8:   add    $0x8,%ecx
  0x0000020ce01409ab:   jmp    0x0000020ce01409b5
  0x0000020ce01409ad:   mov    %rdi,%rbx
  0x0000020ce01409b0:   jmp    0x0000020ce01409b5
  0x0000020ce01409b2:   add    $0x10,%ecx
  0x0000020ce01409b5:   vmovss %xmm0,(%rsp)
  0x0000020ce01409ba:   vmovdqu %xmm3,%xmm0
  0x0000020ce01409be:   cmp    %r10,%rbx
  0x0000020ce01409c1:   mov    $0xffffffff,%ebp
  0x0000020ce01409c6:   jl     0x0000020ce01409d0
  0x0000020ce01409c8:   setne  %bpl
  0x0000020ce01409cc:   movzbl %bpl,%ebp
  0x0000020ce01409d0:   mov    $0xffffff45,%edx
  0x0000020ce01409d5:   mov    %ecx,0x8(%rsp)
  0x0000020ce01409d9:   mov    %r9,0x18(%rsp)
  0x0000020ce01409de:   mov    %r11,0x20(%rsp)
  0x0000020ce01409e3:   vmovq  %xmm0,0x28(%rsp)             ;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce01409e9:   data16 xchg %ax,%ax
  0x0000020ce01409ec:   vzeroupper 
  0x0000020ce01409ef:   call   0x0000020cdfa04600           ; ImmutableOopMap {[24]=Oop [32]=Oop }
                                                            ;*ifgt {reexecute=1 rethrow=0 return_oop=0}
                                                            ; - (reexecute) jdk.incubator.vector.VectorIntrinsics::indexInRange@12 (line 53)
                                                            ; - jdk.incubator.vector.FloatVector::fromArray@17 (line 2829)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@32 (line 44)
                                                            ;   {runtime_call UncommonTrapBlob}
  0x0000020ce01409f4:   nopl   0x564(%rax,%rax,1)           ;   {other}
  0x0000020ce01409fc:   add    $0x10,%ecx
  0x0000020ce01409ff:   add    $0x8,%ecx
  0x0000020ce0140a02:   jmp    0x0000020ce0140a0c
  0x0000020ce0140a04:   mov    %rdi,%rbx
  0x0000020ce0140a07:   jmp    0x0000020ce0140a0c
  0x0000020ce0140a09:   add    $0x10,%ecx
  0x0000020ce0140a0c:   vmovdqu %ymm1,0x20(%rsp)
  0x0000020ce0140a12:   vmovss %xmm0,(%rsp)
  0x0000020ce0140a17:   vmovdqu %xmm3,%xmm0
  0x0000020ce0140a1b:   cmp    %r8,%rbx
  0x0000020ce0140a1e:   mov    $0xffffffff,%ebp
  0x0000020ce0140a23:   jl     0x0000020ce0140a2d
  0x0000020ce0140a25:   setne  %bpl
  0x0000020ce0140a29:   movzbl %bpl,%ebp
  0x0000020ce0140a2d:   mov    $0xffffff45,%edx
  0x0000020ce0140a32:   mov    %ecx,0x8(%rsp)
  0x0000020ce0140a36:   mov    %r11,0x10(%rsp)
  0x0000020ce0140a3b:   mov    %r9,0x40(%rsp)
  0x0000020ce0140a40:   vmovq  %xmm0,0x48(%rsp)             ;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce0140a46:   xchg   %ax,%ax
  0x0000020ce0140a48:   vzeroupper 
  0x0000020ce0140a4b:   call   0x0000020cdfa04600           ; ImmutableOopMap {[16]=Oop [64]=Oop }
                                                            ;*ifgt {reexecute=1 rethrow=0 return_oop=0}
                                                            ; - (reexecute) jdk.incubator.vector.VectorIntrinsics::indexInRange@12 (line 53)
                                                            ; - jdk.incubator.vector.FloatVector::fromArray@17 (line 2829)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@44 (line 45)
                                                            ;   {runtime_call UncommonTrapBlob}
  0x0000020ce0140a50:   nopl   0x10005c0(%rax,%rax,1)       ;   {other}
  0x0000020ce0140a58:   vmovss %xmm0,(%rsp)
  0x0000020ce0140a5d:   vmovdqu %xmm3,%xmm0
  0x0000020ce0140a61:   jmp    0x0000020ce01409be
  0x0000020ce0140a66:   vmovdqu %ymm1,0x20(%rsp)
  0x0000020ce0140a6c:   vmovss %xmm0,(%rsp)
  0x0000020ce0140a71:   vmovdqu %xmm3,%xmm0
  0x0000020ce0140a75:   jmp    0x0000020ce0140a1b
  0x0000020ce0140a77:   vxorps %xmm0,%xmm0,%xmm0
  0x0000020ce0140a7b:   jmp    0x0000020ce01409b5
  0x0000020ce0140a80:   vxorps %xmm0,%xmm0,%xmm0
  0x0000020ce0140a84:   jmp    0x0000020ce0140a0c
  0x0000020ce0140a86:   mov    %r8,%r9
  0x0000020ce0140a89:   mov    $0xffffff76,%edx
  0x0000020ce0140a8e:   mov    %r11,%rbp
  0x0000020ce0140a91:   mov    %r9,(%rsp)                   ;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
  0x0000020ce0140a95:   data16 xchg %ax,%ax
  0x0000020ce0140a98:   vzeroupper 
  0x0000020ce0140a9b:   call   0x0000020cdfa04600           ; ImmutableOopMap {rbp=Oop [0]=Oop }
                                                            ;*if_icmpge {reexecute=1 rethrow=0 return_oop=0}
                                                            ; - (reexecute) DistanceBenchmark$Algorithm::simdL2@8 (line 42)
                                                            ;   {runtime_call UncommonTrapBlob}
  0x0000020ce0140aa0:   nopl   0x2000610(%rax,%rax,1)       ;*invokestatic reductionCoerced {reexecute=0 rethrow=0 return_oop=0}
                                                            ; - jdk.incubator.vector.FloatVector::reduceLanesTemplate@78 (line 2644)
                                                            ; - jdk.incubator.vector.Float256Vector::reduceLanes@2 (line 324)
                                                            ; - DistanceBenchmark$Algorithm::simdL2@69 (line 47)
                                                            ;   {other}
  0x0000020ce0140aa8:   movabs $0x20ce0140997,%r10          ;   {internal_word}
  0x0000020ce0140ab2:   mov    %r10,0x468(%r15)
  0x0000020ce0140ab9:   jmp    0x0000020cdfa05700           ;   {runtime_call SafepointBlob}
  0x0000020ce0140abe:   call   Stub::nmethod_entry_barrier  ;   {runtime_call StubRoutines (final stubs)}
  0x0000020ce0140ac3:   jmp    0x0000020ce014065a
[Exception Handler]
  0x0000020ce0140ac8:   jmp    0x0000020cdfacdd00           ;   {no_reloc}
[Deopt Handler Code]
  0x0000020ce0140acd:   call   0x0000020ce0140ad2
  0x0000020ce0140ad2:   subq   $0x5,(%rsp)
  0x0000020ce0140ad7:   jmp    0x0000020cdfa049a0           ;   {runtime_call DeoptimizationBlob}
  0x0000020ce0140adc:   hlt    
  0x0000020ce0140add:   hlt    
  0x0000020ce0140ade:   hlt    
  0x0000020ce0140adf:   hlt    
--------------------------------------------------------------------------------
[/Disassembly]
```
</details>
