# JavaでのSIMD利用について

JDK 21 の時点で2つの方法がある。

* HotSpot JavaVMの最適化でSIMDを使う
* jdk.incubator.vector を使う
* JavaからGPU

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
