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
