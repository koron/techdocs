# PaliGemmaモデルのfinetuneに特化した検証

以下を検証する

* 他のLLMに比べて層数が少ないのでは?
* PaliGemmaの層の構造はどうなってる?
* finetuneに使ったアテンションって実際はどのあたり?
* imageレイヤーはVision Transformerなの?
* SigLIP, CLIP の位置付けは?

## 他のLLMに比べて層数が少ないのでは?

[実験ノート2](https://github.com/koron/techdocs/blob/main/finetune-paligemma/02_finetune_2nd.ipynb) を見ると層数はわずかに32。

<details>
<summary></summary>

```
img/Transformer/encoder_norm/bias                                                (1152,)                float16
img/Transformer/encoder_norm/scale                                               (1152,)                float16
img/Transformer/encoderblock/LayerNorm_0/bias                                    (27, 1152)             float16
img/Transformer/encoderblock/LayerNorm_0/scale                                   (27, 1152)             float16
img/Transformer/encoderblock/LayerNorm_1/bias                                    (27, 1152)             float16
img/Transformer/encoderblock/LayerNorm_1/scale                                   (27, 1152)             float16
img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias                             (27, 4304)             float16
img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel                           (27, 1152, 4304)       float16
img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias                             (27, 1152)             float16
img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel                           (27, 4304, 1152)       float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias             (27, 16, 72)           float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel           (27, 1152, 16, 72)     float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias             (27, 1152)             float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel           (27, 16, 72, 1152)     float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias           (27, 16, 72)           float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel         (27, 1152, 16, 72)     float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias           (27, 16, 72)           float16
img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel         (27, 1152, 16, 72)     float16
img/embedding/bias                                                               (1152,)                float16
img/embedding/kernel                                                             (14, 14, 3, 1152)      float16
img/head/bias                                                                    (2048,)                float16
img/head/kernel                                                                  (1152, 2048)           float16
img/pos_embedding                                                                (1, 256, 1152)         float16
llm/embedder/input_embedding                                                     (257152, 2048)         float16
llm/final_norm/scale                                                             (2048,)                float16
llm/layers/attn/attn_vec_einsum/w                                                (18, 8, 256, 2048)     float32
llm/layers/attn/kv_einsum/w                                                      (18, 2, 1, 2048, 256)  float32
llm/layers/attn/q_einsum/w                                                       (18, 8, 2048, 256)     float32
llm/layers/mlp/gating_einsum                                                     (18, 2, 2048, 16384)   float16
llm/layers/mlp/linear                                                            (18, 16384, 2048)      float16
llm/layers/pre_attention_norm/scale                                              (18, 2048)             float16
llm/layers/pre_ffw_norm/scale                                                    (18, 2048)             float16
```
</details>

アテンションレイヤーが4次や5次のテンソルになってることから、そこでマルチヘッドを表現して層数が少なくなってる可能性がある。

HF Transformers用のモデルを見ればまた別の視座が得られるかも。
HFで[レイヤー詳細](https://huggingface.co/google/paligemma-3b-pt-224/tree/main?show_file_info=model.safetensors.index.json)を見るとLLM側には0～17の合計18層がある。
やはりマルチヘッドの表現がJAXとHF Transfomers(PyTorch?)では異なるかもしれない。

参考: [PyTorchの全レイヤープロパティ](./pytorch-paligemma-structure.txt)

以下は両LLMに対して推定されるレイヤー名の対応関係。2048を8x256で表現しているところがありそうだ。

JAX側レイヤー名                     | PyTorch側レイヤー名
------------------------------------|--------------------------------------
`pre_attention_norm/scale`          | `input_layernorm`
`pre_ffw_norm/scale`                | `post_attention_layernorm`
`mlp/linear`                        | `mlp.down_proj`
`mlp/gating_einsum`                 | `mlp.gate_proj` <br> `mlp.up_proj`
`attn/q_einsum/w`                   | `self_attn.q_proj`
`attn/attn_vec_einsum/w`            | `self_attn.o_proj`
`attn/attn_vec_einsum/w`            | `self_attn.k_proj` <br> `self_attn.v_proj`

PyTorch側の `multi_modal_projector` の `bias` と `weight` はそれぞれ JAX 側の `img/head/bias`, `img/head/weight` に対応していそう。
名前からしてマルチモーダルの接続点≒imgエンコーダーをllmデコーダーに接続する場所だと考えられる。

PyTorchにおける層はJAXではテンソルに内包されることが分かった。
またVision EncoderとLLMとの接続点と思わしき層を見つけた。

## PaliGemmaの層の構造はどうなってる?

[big\_vision](https://github.com/google-research/big_vision/) の paligemma、
もしくは HF Transformer の `PaliGemmaForConditionalGeneration` を見れば良さそう。
以下はその結論に至った参考情報。

```python
# Import model definition from big_vision
from big_vision.models.proj.paligemma import paligemma
from big_vision.trainers.proj.paligemma import predict_fns
```

<https://huggingface.co/google/paligemma-3b-pt-224/blob/a2bce053c060e22750007165c7919d7dcc9507d2/config.json#L4>

big\_vision で見るべきは [ココ](https://github.com/google-research/big_vision/blob/46b2456f54b9d4f829d1925b78943372b376153d/big_vision/models/proj/paligemma/paligemma.py#L261)

トークン化したテキストを、LLMのembeddingを得るところを利用してembedding。

    [t0, t1, t2, ..., tZ] + ([0]) -> [et0, et1, et2, ..., etZ]

画像をViTを使ってembedding。

    [i0, i1, i2, ..., iZ] -> [ei0, ei1, ei2, ..., eiZ]

2つのembeddedを合体して、

    [et0, et1, et2, ..., etZ] + [ei0, ei1, ei2, ..., eiZ] -> [et0, et1, et2, ..., etZ, ei0, ei1, ei2, ..., eiZ]

LLMでトークン化しテキストに戻す。

    [et0, et1, et2, ..., etZ, ei0, ei1, ei2, ..., eiZ] + [0...] -> [t0, t1, t2, ..., tZ]

## imageレイヤーはVision Transformerなの?

[ココ](https://github.com/google-research/big_vision/blob/46b2456f54b9d4f829d1925b78943372b376153d/big_vision/models/proj/paligemma/paligemma.py#L277) に `vit` の表示があるので Vision Transformer である可能性が高い。
