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

* LoRAで学習してみる (いまのところ学習には失敗している)

    [lora-training.ipynb](./lora-trainging.ipynb)


## Finetune について考えを更新

finetuneは事前学習に対して事後学習くらいの意味合いのようだ。
方式はいくつもあるが大きく分けて、学習済みモデル自体を変更する方法と外付けのアダプターで学習する方法(PEFT)がある。
paligemmaの例は前者で、LoRA始めアダプターと呼ばれるものは後者という理解で良さそう。
モデル自体を変更する方法はどの層までの変更を許すのかでバリエーションがあり、
アダプターを用いる方法ではどのような形のアダプターをどこに付けるのかでバリエーションがある。

## わかってないことば

* attention とは? (transformerの文脈に限定したほうが良いかも)

    * [【文系でもわかる】ChatGPTのキモ「Transformer」「Attention」のしくみ](https://www.sbbit.jp/article/cont1/114721)

    現時点の理解: 文章をトークン化したトークンシーケンスに対して、ここのトークン≒単語が重要だよという重みを付けて学習・推論すること

* transformer における q\_proj などの層名(モジュール名?)の名前付けと、違い

    例として [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it/tree/main) に見られるレイヤーの構造は以下の通り。
    q, k v, o は transformer 用語の Query, Key, Value, Output といったところか?

    ```
    input_layernorm
    mlp.down_proj
    mlp.gate_proj
    mlp.up_proj
    post_attention_layernorm
    post_feedforward_layernorm
    pre_feedforward_layernorm
    self_attn.k_proj
    self_attn.o_proj
    self_attn.q_proj
    self_attn.v_proj
    ```

    * [Python(PyTorch)で自作して理解するTransformer](https://zenn.dev/yukiyada/articles/59f3b820c52571)

    名前等からおおよその対応はわかるが、 transformer の構造をしっかり押さえないとこの先は踏み入れられなさそう。

* LoRAの(ハイパー)パラメーターの意味

    * [【ローカルLLM】QLoRAの複雑なパラメータを（少しでも）整理する](https://note.com/bakushu/n/ne7760c47e39e)
* QLoRAってなんだ?

    量子化したLoRAのこと。
    量子化しているので、モデルの方も量子化する必要がある。
    HFの例では、モデル読み込み時に量子化を指定する。

    <https://gigazine.net/news/20230603-qlora-finetuning-llm/>
