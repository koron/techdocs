# JavaでのSIMD利用について

JDK 21 の時点で2つの方法がある。

* HotSpot JavaVMの最適化でSIMDを使う
* java.incubator.vector を使う

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

### 考察

* byte(8ビットデータ)にすると約10倍というのは想像以上
* 128ビットレジスタを使っているので…
    * floatは理論値で128/32で4倍 (実倍率 2.4)
    * byteは理論値で128/8で16倍 (実倍率 9.5)
    * オーバーヘッドを考えれば実倍率は極めて妥当
* CUDAを使う方法、あったりしない?
    * 数値計算専門ライブラリも調べたほうが良さそう