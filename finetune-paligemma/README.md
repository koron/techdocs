# Finetune PaliGemma

## Resources

* [Video: PaliGemma by Google: Train Model on Custom Detection Dataset](https://www.youtube.com/watch?v=OMBmVInx68M)
    * [Blog: How to Fine-tune PaliGemma for Object Detection Tasks](https://blog.roboflow.com/how-to-fine-tune-paligemma/)

    PaliGemmaを用いて手書き文字を識別(クラスタリング)する精度を上げるfinetuneのチュートリアル。
    Collab上でモデルをダウンロードするためのAPI KEYを払い出すところからの丁寧な内容。
    T4 GPUでVRAM 16GB、RAM 12GBで動かす想定。

    finetune用のデータセットはJSONLで供給。
    各行の要素にはimage, prefix, suffixの3つのプロパティが必要。
    (このあたりはどっかに正規のドキュメントがありそう)
    imageプロパティは画像のパス、prefixプロパティはプロンプト、suffixは想定するレスポンス。
    prefixとsuffixの作り方はタスクによってお作法がある。
    とりわけ物体検出の場合prefixには `detect` というキーワードが、
    suffixには4つの `<loc ...>` タグで与えたバウンディングボックスと物体名が必要となる。
    セミコロンで複数要素を区切れる。

    単一文字(数字+演算子)の識別をfinetuneした。
    式(複数文字)の識別には当然失敗する。
    複数のトランプを識別するfinetuneしたら、できるようになる。

* [bfloat numerical format (Japanese)](https://cloud.google.com/tpu/docs/bfloat16?hl=ja)

    Brain Floating Point Format: 1bit for sign, 8bits for exponent, 7bits mantissa. For Cloud TPU

* [Fine-tune PaliGemma with JAX ](https://ai.google.dev/gemma/docs/paligemma/fine-tuning-paligemma)

    Googleの公式ドキュメント

    JAXが要る。CUDAはLinuxのみ。CUDA使うにはcuDNN, NCCLが要る。
    [NVIDIA/JAX-Toolbox](https://github.com/NVIDIA/JAX-Toolbox) が提供するDockerイメージ使った方が楽かも。

    T4 GPUに収めるためデータを少なくし、attention層だけをfinetuneする。残りの層は freeze。

    [google-research/big_vision](https://github.com/google-research/big_vision) を使う。

## Try with docker

ホスト側は RTX 4070 (12GB RAM)+ Core i9 9900K (64GB RAM) の Windows 11  
Docker Desktop 利用  
MSYS2 の bash 上で作業

ホスト側で以下を実行。
`-gpus all` でコンテナ側にGPUを利用可能に。
`--shm-size=12g` でGPU側と大きなデータのやり取りをできるように。

```console
$ docker --gpus all run --shm-size=12g --rm -it -p 8888:8888 ghcr.io/nvidia/jax:gemma
```



Docker コンテナ内で以下を実行

```console
# pip install gsutil
# cd /opt/gemma/examples
# jupyter lab --allow-root --ip 0.0.0.0
```

表示されたURLをブラウザで表示

`Fineture_Paligemma.ipynb` を表示

`KAGGLE_USERNAME` と `KAGGLE_KEY` に自前の値を設定。

モデルのダウンロードコードを以下の通りに変更

```python
MODEL_PATH = kagglehub.model_download('google/paligemma/jax/paligemma-3b-pt-224', 'paligemma-3b-pt-224.f16.npz')
```

`MODEL_PATH` の不要な上書きをコメントアウト

```python
#MODEL_PATH = "./pt_224.npz"
```

次のエラーで止まった…

```
2024-06-12 04:21:56.566447: W external/tsl/tsl/framework/bfc_allocator.cc:291] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1012.65MiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
---------------------------------------------------------------------------
XlaRuntimeError                           Traceback (most recent call last)
Input In [15], in <cell line: 23>()
     31 # Training step and report training loss
     32 learning_rate = sched_fn(step)
---> 33 params, loss = update_fn(params, batch, learning_rate)
     35 loss = jax.device_get(loss)
     36 print(f"step: {step:2d}/{TRAIN_STEPS:2d}   lr: {learning_rate:.5f}   loss: {loss:.4f}")

    [... skipping hidden 15 frame]

File /opt/jax/jax/_src/compiler.py:238, in backend_compile(backend, module, options, host_callbacks)
    233   return backend.compile(built_c, compile_options=options,
    234                          host_callbacks=host_callbacks)
    235 # Some backends don't have `host_callbacks` option yet
    236 # TODO(sharadmv): remove this fallback when all backends allow `compile`
    237 # to take in `host_callbacks`
--> 238 return backend.compile(built_c, compile_options=options)

XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1061842944 bytes.
```

[Googleのサンプル](https://ai.google.dev/gemma/docs/paligemma/fine-tuning-paligemma) を Colab の T4 GPU (VRAM 15GB, RAM 12 GB) で試したところ完走できた。
ローカルの RTX 4070 の VRAM 12GB では足りない可能性が高くなった。
Colab で実行中にVRAMを監視したところ 11.2GB 前後であったため、
重みの更新時に更新レイヤーの確保に失敗していると推測できる。

学習のバッチサイズを下げることで、実行できそう。
できた!

```
BATCH_SIZE = 2
```

training dataは少し詳しい解説になっている。
まず概要を説明させ、主に色に着目して解説させ、
最後に全体の位置関係やその他のことについて言及している。

学習前にvalidation dataをPaliGemmaへ食わせた。
かなりあっさり目の解説になった。

学習時にはややおかしな回答がまざったりもしたが、
最初の確認からそれっぽい構成の文章が出ていた。

学習後にvalidation dataを食わせた。
training dataに近い文章が出てくるようになっていた。
