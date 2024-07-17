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

[実際のノートブック](./01_run_anyway.ipynb)

## Try with another dataset

先に紹介した <https://blog.roboflow.com/how-to-fine-tune-paligemma/> にある数字・記号識別のfinetuneをColabで試してみる。
[参照: Colabのノートブック(オリジナル)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-paligemma-on-detection-dataset.ipynb?ref=blog.roboflow.com)

roboflowの **Private API Key** が必要なことに注意。

学習前は書かれた図形を必要以上に分解して、複数の字形を識別してしまう。

使ってるデータセット <https://universe.roboflow.com/roboflow-jvuqo/number-ops-j1426/dataset/1> 。
約4900毎のトレーニングセット、約1600枚の検証セット、約600枚のテストセット。

学習後は、1画像に付き1つを認識したことを確認できた。

finetuneしたモデルのダウンロードもできるようだが、大きいせいかうまくいかなかった。
時間がかかってただけかもしれない。

### PaliGemmaのFinetune済モデルに使われたデータセットを見てみる


* [AI2 Diagram Dataset](https://allenai.org/data/diagrams)

    図を元にした設問と回答のデータセット

* [A-OKVQA](https://allenai.org/project/a-okvqa/home)

    回答に多様な知識を必要とする、画像と設問のデータセット(VQA)
    回答には "MC (Multiple-Choice) Answers" と "Direct Answers" の区別がある。

* [COCO-35L](https://arxiv.org/pdf/2205.12522)

    3600枚の画像に対する36ヵ国語によるアノテーション済みデータ

* [COCO Captions](https://cocodataset.org/#overview)

    Common Object in Context のキャプション付きデータ
    実体 <https://cocodataset.org/#captions-2015> かな?

* [DocVQA](https://www.docvqa.org/)

    目的主導で、人間が定義する高次タスクを行えるようにする

* [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html)

    シーングラフに対するVQAを指向している。
    既存データセットが偏っててベンチマークに向かないことに対するアンチテーゼ

* [InfographicVQA](https://openaccess.thecvf.com/content/WACV2022/papers/Mathew_InfographicVQA_WACV_2022_paper.pdf)

    インフォグラフィックから知識を取り出すVQA

* [NLVR](https://lil.nlp.cornell.edu/nlvr/)

    画像に対して与えられた文の真偽を判定するタスク。
    データセット自体は後継があるのでobsolete

* [OCR-VQA](https://ocr-vqa.github.io/)

    画像内のテキストを読んだ上でのVQA

* [OK-VQA](https://okvqa.allenai.org/)

    前出のA-OKVQAの前身

* [RefCOCO](https://arxiv.org/abs/1608.00272)

    ???

* [RSVQA-HR](https://zenodo.org/records/6344367)

    リモートセンシング専門のVQA。高解像版

* [RSVQA-LR](https://zenodo.org/records/6344367)

    リモートセンシング専門のVQA。低解像版

* [SciCap](https://arxiv.org/abs/2110.11624)

    科学的な図に対するキャプションの生成

* [ScienceQA](https://scienceqa.github.io/)

    マルチモーダルな推論で科学の質問に答えるVQA

* [Screen2Words](https://arxiv.org/abs/2108.03353)

    モバイルUIの要約(文章)を生成する

* [SceneTextVQA](https://arxiv.org/abs/1905.13648)

    画像内のテキストに対するVQA

* [TallyQA](https://arxiv.org/abs/1810.12440)

    オブジェクト間の関係性がある計数(数え上げ)のデータセット

* [TextCaps](https://textvqa.org/textcaps/)

    画像内のテキストを読み取って適切なキャプションを付けたデータセット

* [TextVQA](https://textvqa.org/)

    画像内のテキストを読んで推論するVQA

* [VizWizVQA](https://vizwiz.org/tasks-and-datasets/vqa/)

    盲人の質問に答えるVQA

* [VQA](https://visualqa.org/index.html)

    自由形式のVQA。初出2015年、最終更新2017年と古い

* [Widget Captioning](https://arxiv.org/abs/2010.04295)

    モバイルUIに対するキャプションを生成する

## Kaggle で finetunabble か否かはどういう違いがある?

Finetune済みのモデルを再度Finetuneすると…

利点:

* 精度向上
* 新しいタスクへの転移
* 学習データが少ないケースへの対応

欠点:

* 過学習のリスク
* 計算コスト
* データ品質

## 結論

JAX を使って Google PaLI Gemma のファインチューニングする方法を確認できた。
またファインチューニングするにあたっての前提知識を調べた。
