# HuggingFace の「学習のチュートリアル」をhands-on

* [Fine-tune a pretrained model](https://huggingface.co/docs/transformers/ja/training)

    * [Trainer を使ったやつ](./finetune-tutorial.ipynb)
    * [ネィティブPyTorchを使ったやつ](./finetune-native-pytorch.ipynb)

* [Accelerate を用いた分散学習](https://huggingface.co/docs/transformers/ja/accelerate)

    accelerate を使うと4行の修正で分散GPUに対応できる。
    実際に使うには環境に合わせた細かいコンフィグレーションが要るはずなので、
    もうちょっと作業する必要があるだろう。

* [Load adapters with HF PEFT](https://huggingface.co/docs/transformers/ja/peft)

    LoRA, IA3, AdaLoRA が選べる。

    [PEFTのノート](./finetune-peft.ipynb)


## Finetune について考えを更新

finetuneは事前学習に対して事後学習くらいの意味合いのようだ。
方式はいくつもあるが大きく分けて、学習済みモデル自体を変更する方法と外付けのアダプターで学習する方法(PEFT)がある。
paligemmaの例は前者で、LoRA始めアダプターと呼ばれるものは後者という理解で良さそう。
モデル自体を変更する方法はどの層までの変更を許すのかでバリエーションがあり、
アダプターを用いる方法ではどのような形のアダプターをどこに付けるのかでバリエーションがある。

## わかってないことば

* attention とは? (transformerの文脈に限定したほうが良いかも)

    * [【文系でもわかる】ChatGPTのキモ「Transformer」「Attention」のしくみ](https://www.sbbit.jp/article/cont1/114721)

* transformer における q\_proj などの層名(モジュール名?)の名前付けと、違い
* LoRAのパラメーターの意味
* QLoRAってなんだ?

    量子化したLoRAのこと。
    量子化しているので、モデルの方も量子化する必要がある。
    HFの例では、モデル読み込み時に量子化を指定する。

    <https://gigazine.net/news/20230603-qlora-finetuning-llm/>
