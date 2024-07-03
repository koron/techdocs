# CyberAgentのVLMを触ってみた記録

CyberAgentがApache 2.0ライセンスで公開している日本語Vision Language Model(VLM)を触ってみた際のメモ

* モデルデータ <https://huggingface.co/cyberagent/llava-calm2-siglip>
* プレスリリース <https://www.cyberagent.co.jp/news/detail/id=30344>

## まずは普通に動かす

HuggingFaceからgit (+lfs)でモデルデータを取得。

    git clone https://huggingface.co/cyberagent/llava-calm2-siglip

CUDA 12.2 + Python 3.11 に必要なパッケージをインストール。
PyTorch は <https://pytorch.org/get-started/locally/> 公式サイトのインストラクションに従うと WindowsネイティブPythonでもCUDAが使える。
Windowsネイティブを切ったTensorFlowとは違って、好感度アップ。

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

あとは jupyter lab を起動して [ノートブック: getting-started.ipynb](./getting-started.ipynb) を開いて実行。
モデルデータのページに記載されてる Usage から少しだけ修正している。
画像に対する質問に答える、サンプル通りに実行できた。

    jupyter lab

モデルは 7.5B で bf16 なので 15GB ほど。
TorchはGPUとCPUを併用できるらしく、15GBのモデルは分散して配置された。
大半をGPUに置いた時は推論に2分弱(平均100秒)かかった。
CPUに比重が置かれると5分ほどかかった。

## 量子化を試みる(失敗)

手元の環境で1回の推論に約2分かかるため、[llama.cpp](https://github.com/ggerganov/llama.cpp) での量子化(GGUF)を試みた。
結果、失敗した。

F16でGGUF化したあとにQ8やQ4への量子化をする。

参考にした手順はこちら: <https://github.com/ggerganov/llama.cpp/tree/master/examples/llava#llava-15>

大まかにいうとこんな感じ

1. [examples/llava/llava-surgery-v2.py](https://github.com/ggerganov/llama.cpp/blob/master/examples/llava/llava-surgery-v2.py) でモデルを分離
2. [examples/llava/convert-image-encoder-to-gguf.py](https://github.com/ggerganov/llama.cpp/blob/master/examples/llava/convert-image-encoder-to-gguf.py) で視覚モデルをGGUF化
3. [examples/convert-legacy-llama.py](https://github.com/ggerganov/llama.cpp/blob/master/examples/convert-legacy-llama.py) で言語モデルをGGUF化
4. [examples/quantize](https://github.com/ggerganov/llama.cpp/tree/master/examples/quantize) で量子化

llama.cppのGGUFコンバーターはレイヤーの名前に、モデル依存の強い想定を置いていた。
HF Transformersでそれをどう解決しているのかはわからないが、
少し手を入れることでコンバートだけはできた。

ただしそうやってできたGGUFはllama.cppでの動作が怪しい。
なにかしら前提に誤りがあると考えられる。

観測された不具合:

* 言語モデルの回答が不安定
* 視覚モデルが読み込めない
* 変換時のTokenizerの指定が良く分からない

<details>
<summary>llama.cppに当てたパッチ</summary>

```
diff --git a/examples/convert-legacy-llama.py b/examples/convert-legacy-llama.py
index 721a57c0..4fc1ee0e 100755
--- a/examples/convert-legacy-llama.py
+++ b/examples/convert-legacy-llama.py
@@ -187,13 +187,13 @@ class Params:
     @staticmethod
     def guessed(model: LazyModel) -> Params:
         # try transformer naming first
-        n_vocab, n_embd = model["model.embed_tokens.weight"].shape if "model.embed_tokens.weight" in model else model["tok_embeddings.weight"].shape
+        n_vocab, n_embd = model["language_model.model.embed_tokens.weight"].shape if "language_model.model.embed_tokens.weight" in model else model["tok_embeddings.weight"].shape
 
         # try transformer naming first
-        if "model.layers.0.self_attn.q_proj.weight" in model:
-            n_layer = next(i for i in itertools.count() if f"model.layers.{i}.self_attn.q_proj.weight" not in model)
-        elif "model.layers.0.self_attn.W_pack.weight" in model:   # next: try baichuan naming
-            n_layer = next(i for i in itertools.count() if f"model.layers.{i}.self_attn.W_pack.weight" not in model)
+        if "language_model.model.layers.0.self_attn.q_proj.weight" in model:
+            n_layer = next(i for i in itertools.count() if f"language_model.model.layers.{i}.self_attn.q_proj.weight" not in model)
+        elif "language_model.model.layers.0.self_attn.W_pack.weight" in model:   # next: try baichuan naming
+            n_layer = next(i for i in itertools.count() if f"language_model.model.layers.{i}.self_attn.W_pack.weight" not in model)
         else:
             n_layer = next(i for i in itertools.count() if f"layers.{i}.attention.wq.weight" not in model)
 
@@ -225,6 +225,8 @@ class Params:
     def loadHFTransformerJson(model: LazyModel, config_path: Path) -> Params:
         with open(config_path) as f:
             config = json.load(f)
+        if config["text_config"] is not None:
+            config = config["text_config"]
 
         rope_scaling_type = f_rope_scale = n_ctx_orig = rope_finetuned = None
         rope_scaling = config.get("rope_scaling")
@@ -546,7 +548,7 @@ def merge_multifile_models(models_plus: list[ModelPlus]) -> ModelPlus:
     except StopIteration:
         vocab = None
 
-    if any("model.embed_tokens.weight" in mp.model for mp in models_plus):
+    if any("language_model.model.embed_tokens.weight" in mp.model for mp in models_plus):
         # Transformers models put different tensors in different files, but
         # don't split individual tensors between files.
         model: LazyModel = {}
@@ -984,6 +986,7 @@ class OutputFile:
         of = OutputFile(fname_out, endianess=endianess)
 
         # meta data
+        print(params)
         of.add_meta_model(params, metadata)
         of.add_meta_arch(params)
         if isinstance(vocab, Vocab):
@@ -1071,26 +1074,26 @@ def convert_model_names(model: LazyModel, params: Params, skip_unknown: bool) ->
                     if f"layers.{i_l}.feed_forward.experts.{e}.w{w}.weight" in model:
                         experts.append(model[f"layers.{i_l}.feed_forward.experts.{e}.w{w}.weight"])
                         del tmp[f"layers.{i_l}.feed_forward.experts.{e}.w{w}.weight"]
-                    elif f"model.layers.{i_l}.block_sparse_moe.experts.{e}.w{w}.weight" in model:
-                        experts.append(model[f"model.layers.{i_l}.block_sparse_moe.experts.{e}.w{w}.weight"])
-                        del tmp[f"model.layers.{i_l}.block_sparse_moe.experts.{e}.w{w}.weight"]
+                    elif f"language_model.model.layers.{i_l}.block_sparse_moe.experts.{e}.w{w}.weight" in model:
+                        experts.append(model[f"language_model.model.layers.{i_l}.block_sparse_moe.experts.{e}.w{w}.weight"])
+                        del tmp[f"language_model.model.layers.{i_l}.block_sparse_moe.experts.{e}.w{w}.weight"]
                     else:
                         raise ValueError(f"Expert tensor not found: layers.{i_l}.feed_forward.experts.{e}.w{w}.weight")
                 tmp[f"layers.{i_l}.feed_forward.experts.w{w}.weight"] = pack_experts_lazy(experts)
 
     # HF models permut or pack some of the tensors, so we need to undo that
     for i in itertools.count():
-        if f"model.layers.{i}.self_attn.q_proj.weight" in model:
+        if f"language_model.model.layers.{i}.self_attn.q_proj.weight" in model:
             logger.debug(f"Permuting layer {i}")
-            tmp[f"model.layers.{i}.self_attn.q_proj.weight"] = permute_lazy(model[f"model.layers.{i}.self_attn.q_proj.weight"], params.n_head, params.n_head)
-            tmp[f"model.layers.{i}.self_attn.k_proj.weight"] = permute_lazy(model[f"model.layers.{i}.self_attn.k_proj.weight"], params.n_head, params.n_head_kv)
-            # tmp[f"model.layers.{i}.self_attn.v_proj.weight"] =              model[f"model.layers.{i}.self_attn.v_proj.weight"]
-        elif f"model.layers.{i}.self_attn.W_pack.weight" in model:
+            tmp[f"language_model.model.layers.{i}.self_attn.q_proj.weight"] = permute_lazy(model[f"language_model.model.layers.{i}.self_attn.q_proj.weight"], params.n_head, params.n_head)
+            tmp[f"language_model.model.layers.{i}.self_attn.k_proj.weight"] = permute_lazy(model[f"language_model.model.layers.{i}.self_attn.k_proj.weight"], params.n_head, params.n_head_kv)
+            # tmp[f"language_model.model.layers.{i}.self_attn.v_proj.weight"] =              model[f"language_model.model.layers.{i}.self_attn.v_proj.weight"]
+        elif f"language_model.model.layers.{i}.self_attn.W_pack.weight" in model:
             logger.debug(f"Unpacking and permuting layer {i}")
-            tmp[f"model.layers.{i}.self_attn.q_proj.weight"] = permute_part_lazy(model[f"model.layers.{i}.self_attn.W_pack.weight"], 0, params.n_head, params.n_head)
-            tmp[f"model.layers.{i}.self_attn.k_proj.weight"] = permute_part_lazy(model[f"model.layers.{i}.self_attn.W_pack.weight"], 1, params.n_head, params.n_head_kv)
-            tmp[f"model.layers.{i}.self_attn.v_proj.weight"] = part_lazy        (model[f"model.layers.{i}.self_attn.W_pack.weight"], 2)
-            del tmp[f"model.layers.{i}.self_attn.W_pack.weight"]
+            tmp[f"language_model.model.layers.{i}.self_attn.q_proj.weight"] = permute_part_lazy(model[f"language_model.model.layers.{i}.self_attn.W_pack.weight"], 0, params.n_head, params.n_head)
+            tmp[f"language_model.model.layers.{i}.self_attn.k_proj.weight"] = permute_part_lazy(model[f"language_model.model.layers.{i}.self_attn.W_pack.weight"], 1, params.n_head, params.n_head_kv)
+            tmp[f"language_model.model.layers.{i}.self_attn.v_proj.weight"] = part_lazy        (model[f"language_model.model.layers.{i}.self_attn.W_pack.weight"], 2)
+            del tmp[f"language_model.model.layers.{i}.self_attn.W_pack.weight"]
         else:
             break
 
diff --git a/examples/llava/convert-image-encoder-to-gguf.py b/examples/llava/convert-image-encoder-to-gguf.py
index b00bf7c6..6b1e1e64 100644
--- a/examples/llava/convert-image-encoder-to-gguf.py
+++ b/examples/llava/convert-image-encoder-to-gguf.py
@@ -5,6 +5,10 @@ import re
 
 import torch
 import numpy as np
+if 'NO_LOCAL_GGUF' not in os.environ:
+    import sys
+    from pathlib import Path
+    sys.path.insert(1, str(Path(__file__).parent.parent.parent / 'gguf-py'))
 from gguf import *
 from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
 
@@ -193,7 +197,8 @@ if has_text_encoder:
     fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, TEXT), t_hparams["num_attention_heads"])
     fout.add_float32(k(KEY_ATTENTION_LAYERNORM_EPS, TEXT), t_hparams["layer_norm_eps"])
     fout.add_uint32(k(KEY_BLOCK_COUNT, TEXT), t_hparams["num_hidden_layers"])
-    fout.add_token_list(tokens)
+    if tokens is not None:
+        fout.add_token_list(tokens)
 
 if has_vision_encoder:
     # vision_model hparams
diff --git a/examples/llava/llava-surgery-v2.py b/examples/llava/llava-surgery-v2.py
index eb56d698..595444d5 100644
--- a/examples/llava/llava-surgery-v2.py
+++ b/examples/llava/llava-surgery-v2.py
@@ -85,8 +85,11 @@ def find_relevant_checkpoints(checkpoint_paths, newline_criteria, projector):
 def newline_criteria(checkpoint):
     return any(k.startswith("model.image_newline") for k in checkpoint.keys())
 
+def is_key_mm(k):
+    return k.startswith("model.mm_projector") or k.startswith("vision_proj.") or k.startswith("multi_modal_projector.") or k.startswith("vision_tower.")
+
 def proj_criteria(checkpoint):
-    return any(k.startswith("model.mm_projector") or k.startswith("vision_proj.") for k in checkpoint.keys())
+    return any(is_key_mm(k) for k in checkpoint.keys())
 
 
 # Command-line interface setup
@@ -128,7 +131,7 @@ mm_tensors = []
 last_checkpoint = None
 if projector_checkpoint_path is not None:
     last_checkpoint, file_type = load_model(projector_checkpoint_path)
-    mm_tensors = [k for k, v in last_checkpoint.items() if k.startswith("model.mm_projector") or k.startswith("vision_proj.")]
+    mm_tensors = [k for k, v in last_checkpoint.items() if is_key_mm(k)]
 
 if len(mm_tensors) == 0:
     if last_checkpoint is not None:
diff --git a/gguf-py/gguf/tensor_mapping.py b/gguf-py/gguf/tensor_mapping.py
index 81b4992a..ee5f60b9 100644
--- a/gguf-py/gguf/tensor_mapping.py
+++ b/gguf-py/gguf/tensor_mapping.py
@@ -13,7 +13,7 @@ class TensorNameMap:
             "transformer.wte",                           # gpt2 gpt-j mpt refact qwen dbrx
             "transformer.word_embeddings",               # falcon
             "word_embeddings",                           # bloom
-            "model.embed_tokens",                        # llama-hf
+            "language_model.model.embed_tokens",                        # llama-hf
             "tok_embeddings",                            # llama-pth
             "embeddings.word_embeddings",                # bert nomic-bert
             "language_model.embedding.word_embeddings",  # persimmon
@@ -48,7 +48,7 @@ class TensorNameMap:
         # Output
         MODEL_TENSOR.OUTPUT: (
             "embed_out",                 # gptneox
-            "lm_head",                   # gpt2 mpt falcon llama-hf baichuan qwen mamba dbrx
+            "language_model.lm_head",                   # gpt2 mpt falcon llama-hf baichuan qwen mamba dbrx
             "output",                    # llama-pth bloom internlm2
             "word_embeddings_for_head",  # persimmon
             "lm_head.linear",            # phi2
@@ -58,7 +58,7 @@ class TensorNameMap:
         MODEL_TENSOR.OUTPUT_NORM: (
             "gpt_neox.final_layer_norm",               # gptneox
             "transformer.ln_f",                        # gpt2 gpt-j falcon
-            "model.norm",                              # llama-hf baichuan internlm2
+            "language_model.model.norm",                              # llama-hf baichuan internlm2
             "norm",                                    # llama-pth
             "transformer.norm_f",                      # mpt dbrx
             "ln_f",                                    # refact bloom qwen gpt2
@@ -85,15 +85,15 @@ class TensorNameMap:
             "transformer.h.{bid}.input_layernorm",                  # falcon7b
             "h.{bid}.input_layernorm",                              # bloom
             "transformer.h.{bid}.ln_mlp",                           # falcon40b
-            "model.layers.{bid}.input_layernorm",                   # llama-hf
+            "language_model.model.layers.{bid}.input_layernorm",                   # llama-hf
             "layers.{bid}.attention_norm",                          # llama-pth
             "language_model.encoder.layers.{bid}.input_layernorm",  # persimmon
-            "model.layers.{bid}.ln1",                               # yi
+            "language_model.model.layers.{bid}.ln1",                               # yi
             "h.{bid}.ln_1",                                         # gpt2
             "transformer.h.{bid}.ln",                               # phi2
             "model.layers.layers.{bid}.norm",                       # plamo
-            "model.layers.{bid}.attention_norm",                    # internlm2
-            "model.layers.{bid}.norm",                              # mamba-qbert
+            "language_model.model.layers.{bid}.attention_norm",                    # internlm2
+            "language_model.model.layers.{bid}.norm",                              # mamba-qbert
             "backbone.layers.{bid}.norm",                           # mamba
             "transformer.decoder_layer.{bid}.rms_norm",             # Grok
             "transformer.blocks.{bid}.norm_attn_norm.norm_1",       # dbrx
@@ -114,45 +114,45 @@ class TensorNameMap:
             "transformer.h.{bid}.self_attention.query_key_value",                  # falcon
             "h.{bid}.self_attention.query_key_value",                              # bloom
             "language_model.encoder.layers.{bid}.self_attention.query_key_value",  # persimmon
-            "model.layers.{bid}.self_attn.query_key_value",                        # persimmon
+            "language_model.model.layers.{bid}.self_attn.query_key_value",                        # persimmon
             "h.{bid}.attn.c_attn",                                                 # gpt2
             "transformer.h.{bid}.mixer.Wqkv",                                      # phi2
             "encoder.layers.{bid}.attn.Wqkv",                                      # nomic-bert
-            "model.layers.{bid}.self_attn.qkv_proj"                                # phi3
+            "language_model.model.layers.{bid}.self_attn.qkv_proj"                                # phi3
         ),
 
         # Attention query
         MODEL_TENSOR.ATTN_Q: (
-            "model.layers.{bid}.self_attn.q_proj",                       # llama-hf
+            "language_model.model.layers.{bid}.self_attn.q_proj",                       # llama-hf
             "layers.{bid}.attention.wq",                                 # llama-pth
             "encoder.layer.{bid}.attention.self.query",                  # bert
             "transformer.h.{bid}.attn.q_proj",                           # gpt-j
             "model.layers.layers.{bid}.self_attn.q_proj",                # plamo
-            "model.layers.{bid}.attention.wq",                           # internlm2
+            "language_model.model.layers.{bid}.attention.wq",                           # internlm2
             "transformer.decoder_layer.{bid}.multi_head_attention.query" # Grok
         ),
 
         # Attention key
         MODEL_TENSOR.ATTN_K: (
-            "model.layers.{bid}.self_attn.k_proj",                     # llama-hf
+            "language_model.model.layers.{bid}.self_attn.k_proj",                     # llama-hf
             "layers.{bid}.attention.wk",                               # llama-pth
             "encoder.layer.{bid}.attention.self.key",                  # bert
             "transformer.h.{bid}.attn.k_proj",                         # gpt-j
             "transformer.h.{bid}.attn.k",                              # refact
             "model.layers.layers.{bid}.self_attn.k_proj",              # plamo
-            "model.layers.{bid}.attention.wk",                         # internlm2
+            "language_model.model.layers.{bid}.attention.wk",                         # internlm2
             "transformer.decoder_layer.{bid}.multi_head_attention.key" # Grok
         ),
 
         # Attention value
         MODEL_TENSOR.ATTN_V: (
-            "model.layers.{bid}.self_attn.v_proj",                       # llama-hf
+            "language_model.model.layers.{bid}.self_attn.v_proj",                       # llama-hf
             "layers.{bid}.attention.wv",                                 # llama-pth
             "encoder.layer.{bid}.attention.self.value",                  # bert
             "transformer.h.{bid}.attn.v_proj",                           # gpt-j
             "transformer.h.{bid}.attn.v",                                # refact
             "model.layers.layers.{bid}.self_attn.v_proj",                # plamo
-            "model.layers.{bid}.attention.wv",                           # internlm2
+            "language_model.model.layers.{bid}.attention.wv",                           # internlm2
             "transformer.decoder_layer.{bid}.multi_head_attention.value" # Grok
         ),
 
@@ -163,16 +163,16 @@ class TensorNameMap:
             "transformer.blocks.{bid}.attn.out_proj",                       # mpt
             "transformer.h.{bid}.self_attention.dense",                     # falcon
             "h.{bid}.self_attention.dense",                                 # bloom
-            "model.layers.{bid}.self_attn.o_proj",                          # llama-hf
+            "language_model.model.layers.{bid}.self_attn.o_proj",                          # llama-hf
             "layers.{bid}.attention.wo",                                    # llama-pth
             "encoder.layer.{bid}.attention.output.dense",                   # bert
             "transformer.h.{bid}.attn.out_proj",                            # gpt-j
             "language_model.encoder.layers.{bid}.self_attention.dense",     # persimmon
-            "model.layers.{bid}.self_attn.dense",                           # persimmon
+            "language_model.model.layers.{bid}.self_attn.dense",                           # persimmon
             "h.{bid}.attn.c_proj",                                          # gpt2
             "transformer.h.{bid}.mixer.out_proj",                           # phi2
             "model.layers.layers.{bid}.self_attn.o_proj",                   # plamo
-            "model.layers.{bid}.attention.wo",                              # internlm2
+            "language_model.model.layers.{bid}.attention.wo",                              # internlm2
             "encoder.layers.{bid}.attn.out_proj",                           # nomic-bert
             "transformer.decoder_layer.{bid}.multi_head_attention.linear",  # Grok
             "transformer.blocks.{bid}.norm_attn_norm.attn.out_proj",        # dbrx
@@ -188,7 +188,7 @@ class TensorNameMap:
 
         # Rotary embeddings
         MODEL_TENSOR.ATTN_ROT_EMBD: (
-            "model.layers.{bid}.self_attn.rotary_emb.inv_freq",        # llama-hf
+            "language_model.model.layers.{bid}.self_attn.rotary_emb.inv_freq",        # llama-hf
             "layers.{bid}.attention.inner_attention.rope.freqs",       # llama-pth
             "model.layers.layers.{bid}.self_attn.rotary_emb.inv_freq", # plamo
             "transformer.h.{bid}.attn.rotary_emb.inv_freq",            # codeshell
@@ -200,25 +200,25 @@ class TensorNameMap:
             "transformer.h.{bid}.ln_2",                                      # gpt2 refact qwen
             "h.{bid}.post_attention_layernorm",                              # bloom
             "transformer.blocks.{bid}.norm_2",                               # mpt
-            "model.layers.{bid}.post_attention_layernorm",                   # llama-hf
+            "language_model.model.layers.{bid}.post_attention_layernorm",                   # llama-hf
             "layers.{bid}.ffn_norm",                                         # llama-pth
             "language_model.encoder.layers.{bid}.post_attention_layernorm",  # persimmon
-            "model.layers.{bid}.ln2",                                        # yi
+            "language_model.model.layers.{bid}.ln2",                                        # yi
             "h.{bid}.ln_2",                                                  # gpt2
-            "model.layers.{bid}.ffn_norm",                                   # internlm2
+            "language_model.model.layers.{bid}.ffn_norm",                                   # internlm2
             "transformer.decoder_layer.{bid}.rms_norm_2",                    # Grok
         ),
 
         MODEL_TENSOR.FFN_GATE_INP: (
             "layers.{bid}.feed_forward.gate",             # mixtral
-            "model.layers.{bid}.block_sparse_moe.gate",   # mixtral
-            "model.layers.{bid}.mlp.gate",                # qwen2moe
+            "language_model.model.layers.{bid}.block_sparse_moe.gate",   # mixtral
+            "language_model.model.layers.{bid}.mlp.gate",                # qwen2moe
             "transformer.decoder_layer.{bid}.router",     # Grok
             "transformer.blocks.{bid}.ffn.router.layer",  # dbrx
         ),
 
         MODEL_TENSOR.FFN_GATE_INP_SHEXP: (
-            "model.layers.{bid}.mlp.shared_expert_gate", # qwen2moe
+            "language_model.model.layers.{bid}.mlp.shared_expert_gate", # qwen2moe
         ),
 
         # Feed-forward up
@@ -228,36 +228,36 @@ class TensorNameMap:
             "transformer.blocks.{bid}.ffn.up_proj",                   # mpt
             "transformer.h.{bid}.mlp.dense_h_to_4h",                  # falcon
             "h.{bid}.mlp.dense_h_to_4h",                              # bloom
-            "model.layers.{bid}.mlp.up_proj",                         # llama-hf refact
+            "language_model.model.layers.{bid}.mlp.up_proj",                         # llama-hf refact
             "layers.{bid}.feed_forward.w3",                           # llama-pth
             "encoder.layer.{bid}.intermediate.dense",                 # bert
             "transformer.h.{bid}.mlp.fc_in",                          # gpt-j
             "transformer.h.{bid}.mlp.linear_3",                       # refact
             "language_model.encoder.layers.{bid}.mlp.dense_h_to_4h",  # persimmon
-            "model.layers.{bid}.mlp.dense_h_to_4h",                   # persimmon
+            "language_model.model.layers.{bid}.mlp.dense_h_to_4h",                   # persimmon
             "transformer.h.{bid}.mlp.w1",                             # qwen
             "h.{bid}.mlp.c_fc",                                       # gpt2
             "transformer.h.{bid}.mlp.fc1",                            # phi2
-            "model.layers.{bid}.mlp.fc1",                             # phi2
-            "model.layers.{bid}.mlp.gate_up_proj",                    # phi3
+            "language_model.model.layers.{bid}.mlp.fc1",                             # phi2
+            "language_model.model.layers.{bid}.mlp.gate_up_proj",                    # phi3
             "model.layers.layers.{bid}.mlp.up_proj",                  # plamo
-            "model.layers.{bid}.feed_forward.w3",                     # internlm2
+            "language_model.model.layers.{bid}.feed_forward.w3",                     # internlm2
             "encoder.layers.{bid}.mlp.fc11",                          # nomic-bert
-            "model.layers.{bid}.mlp.c_fc",                            # starcoder2
+            "language_model.model.layers.{bid}.mlp.c_fc",                            # starcoder2
             "encoder.layer.{bid}.mlp.gated_layers_v",                 # jina-bert-v2
-            "model.layers.{bid}.residual_mlp.w3",                     # arctic
+            "language_model.model.layers.{bid}.residual_mlp.w3",                     # arctic
         ),
 
         MODEL_TENSOR.FFN_UP_EXP: (
             "layers.{bid}.feed_forward.experts.w3",          # mixtral (merged)
             "transformer.decoder_layer.{bid}.moe.linear_v",  # Grok (merged)
             "transformer.blocks.{bid}.ffn.experts.mlp.v1",   # dbrx
-            "model.layers.{bid}.mlp.experts.up_proj",        # qwen2moe (merged)
+            "language_model.model.layers.{bid}.mlp.experts.up_proj",        # qwen2moe (merged)
         ),
 
         MODEL_TENSOR.FFN_UP_SHEXP: (
-            "model.layers.{bid}.mlp.shared_expert.up_proj",  # qwen2moe
-            "model.layers.{bid}.mlp.shared_experts.up_proj", # deepseek2
+            "language_model.model.layers.{bid}.mlp.shared_expert.up_proj",  # qwen2moe
+            "language_model.model.layers.{bid}.mlp.shared_experts.up_proj", # deepseek2
         ),
 
         # AWQ-activation gate
@@ -267,27 +267,27 @@ class TensorNameMap:
 
         # Feed-forward gate
         MODEL_TENSOR.FFN_GATE: (
-            "model.layers.{bid}.mlp.gate_proj",           # llama-hf refact
+            "language_model.model.layers.{bid}.mlp.gate_proj",           # llama-hf refact
             "layers.{bid}.feed_forward.w1",               # llama-pth
             "transformer.h.{bid}.mlp.w2",                 # qwen
             "model.layers.layers.{bid}.mlp.gate_proj",    # plamo
-            "model.layers.{bid}.feed_forward.w1",         # internlm2
+            "language_model.model.layers.{bid}.feed_forward.w1",         # internlm2
             "encoder.layers.{bid}.mlp.fc12",              # nomic-bert
             "encoder.layer.{bid}.mlp.gated_layers_w",     # jina-bert-v2
             "transformer.h.{bid}.mlp.linear_1",           # refact
-            "model.layers.{bid}.residual_mlp.w1",         # arctic
+            "language_model.model.layers.{bid}.residual_mlp.w1",         # arctic
         ),
 
         MODEL_TENSOR.FFN_GATE_EXP: (
             "layers.{bid}.feed_forward.experts.w1",         # mixtral (merged)
             "transformer.decoder_layer.{bid}.moe.linear",   # Grok (merged)
             "transformer.blocks.{bid}.ffn.experts.mlp.w1",  # dbrx
-            "model.layers.{bid}.mlp.experts.gate_proj",     # qwen2moe (merged)
+            "language_model.model.layers.{bid}.mlp.experts.gate_proj",     # qwen2moe (merged)
         ),
 
         MODEL_TENSOR.FFN_GATE_SHEXP: (
-            "model.layers.{bid}.mlp.shared_expert.gate_proj",  # qwen2moe
-            "model.layers.{bid}.mlp.shared_experts.gate_proj", # deepseek2
+            "language_model.model.layers.{bid}.mlp.shared_expert.gate_proj",  # qwen2moe
+            "language_model.model.layers.{bid}.mlp.shared_experts.gate_proj", # deepseek2
         ),
 
         # Feed-forward down
@@ -297,21 +297,21 @@ class TensorNameMap:
             "transformer.blocks.{bid}.ffn.down_proj",                 # mpt
             "transformer.h.{bid}.mlp.dense_4h_to_h",                  # falcon
             "h.{bid}.mlp.dense_4h_to_h",                              # bloom
-            "model.layers.{bid}.mlp.down_proj",                       # llama-hf
+            "language_model.model.layers.{bid}.mlp.down_proj",                       # llama-hf
             "layers.{bid}.feed_forward.w2",                           # llama-pth
             "encoder.layer.{bid}.output.dense",                       # bert
             "transformer.h.{bid}.mlp.fc_out",                         # gpt-j
             "language_model.encoder.layers.{bid}.mlp.dense_4h_to_h",  # persimmon
-            "model.layers.{bid}.mlp.dense_4h_to_h",                   # persimmon
+            "language_model.model.layers.{bid}.mlp.dense_4h_to_h",                   # persimmon
             "h.{bid}.mlp.c_proj",                                     # gpt2
             "transformer.h.{bid}.mlp.fc2",                            # phi2
-            "model.layers.{bid}.mlp.fc2",                             # phi2
+            "language_model.model.layers.{bid}.mlp.fc2",                             # phi2
             "model.layers.layers.{bid}.mlp.down_proj",                # plamo
-            "model.layers.{bid}.feed_forward.w2",                     # internlm2
+            "language_model.model.layers.{bid}.feed_forward.w2",                     # internlm2
             "encoder.layers.{bid}.mlp.fc2",                           # nomic-bert
-            "model.layers.{bid}.mlp.c_proj",                          # starcoder2
+            "language_model.model.layers.{bid}.mlp.c_proj",                          # starcoder2
             "encoder.layer.{bid}.mlp.wo",                             # jina-bert-v2
-            "model.layers.{bid}.residual_mlp.w2",                     # arctic
+            "language_model.model.layers.{bid}.residual_mlp.w2",                     # arctic
             "encoder.layer.{bid}.mlp.down_layer",                     # jina-bert-v2
         ),
 
@@ -319,26 +319,26 @@ class TensorNameMap:
             "layers.{bid}.feed_forward.experts.w2",          # mixtral (merged)
             "transformer.decoder_layer.{bid}.moe.linear_1",  # Grok (merged)
             "transformer.blocks.{bid}.ffn.experts.mlp.w2",   # dbrx
-            "model.layers.{bid}.mlp.experts.down_proj",      # qwen2moe (merged)
+            "language_model.model.layers.{bid}.mlp.experts.down_proj",      # qwen2moe (merged)
         ),
 
         MODEL_TENSOR.FFN_DOWN_SHEXP: (
-            "model.layers.{bid}.mlp.shared_expert.down_proj",  # qwen2moe
-            "model.layers.{bid}.mlp.shared_experts.down_proj", # deepseek2
+            "language_model.model.layers.{bid}.mlp.shared_expert.down_proj",  # qwen2moe
+            "language_model.model.layers.{bid}.mlp.shared_experts.down_proj", # deepseek2
         ),
 
         MODEL_TENSOR.ATTN_Q_NORM: (
             "language_model.encoder.layers.{bid}.self_attention.q_layernorm",
-            "model.layers.{bid}.self_attn.q_layernorm",                       # persimmon
-            "model.layers.{bid}.self_attn.q_norm",                            # cohere
+            "language_model.model.layers.{bid}.self_attn.q_layernorm",                       # persimmon
+            "language_model.model.layers.{bid}.self_attn.q_norm",                            # cohere
             "transformer.blocks.{bid}.attn.q_ln",                             # sea-lion
             "encoder.layer.{bid}.attention.self.layer_norm_q"                 # jina-bert-v2
         ),
 
         MODEL_TENSOR.ATTN_K_NORM: (
             "language_model.encoder.layers.{bid}.self_attention.k_layernorm",
-            "model.layers.{bid}.self_attn.k_layernorm",                       # persimmon
-            "model.layers.{bid}.self_attn.k_norm",                            # cohere
+            "language_model.model.layers.{bid}.self_attn.k_layernorm",                       # persimmon
+            "language_model.model.layers.{bid}.self_attn.k_norm",                            # cohere
             "transformer.blocks.{bid}.attn.k_ln",                             # sea-lion
             "encoder.layer.{bid}.attention.self.layer_norm_k"                 # jina-bert-v2
         ),
@@ -356,62 +356,62 @@ class TensorNameMap:
         ),
 
         MODEL_TENSOR.SSM_IN: (
-            "model.layers.{bid}.in_proj",
+            "language_model.model.layers.{bid}.in_proj",
             "backbone.layers.{bid}.mixer.in_proj",
         ),
 
         MODEL_TENSOR.SSM_CONV1D: (
-            "model.layers.{bid}.conv1d",
+            "language_model.model.layers.{bid}.conv1d",
             "backbone.layers.{bid}.mixer.conv1d",
         ),
 
         MODEL_TENSOR.SSM_X: (
-            "model.layers.{bid}.x_proj",
+            "language_model.model.layers.{bid}.x_proj",
             "backbone.layers.{bid}.mixer.x_proj",
         ),
 
         MODEL_TENSOR.SSM_DT: (
-            "model.layers.{bid}.dt_proj",
+            "language_model.model.layers.{bid}.dt_proj",
             "backbone.layers.{bid}.mixer.dt_proj",
         ),
 
         MODEL_TENSOR.SSM_A: (
-            "model.layers.{bid}.A_log",
+            "language_model.model.layers.{bid}.A_log",
             "backbone.layers.{bid}.mixer.A_log",
         ),
 
         MODEL_TENSOR.SSM_D: (
-            "model.layers.{bid}.D",
+            "language_model.model.layers.{bid}.D",
             "backbone.layers.{bid}.mixer.D",
         ),
 
         MODEL_TENSOR.SSM_OUT: (
-            "model.layers.{bid}.out_proj",
+            "language_model.model.layers.{bid}.out_proj",
             "backbone.layers.{bid}.mixer.out_proj",
         ),
 
         MODEL_TENSOR.ATTN_Q_A: (
-            "model.layers.{bid}.self_attn.q_a_proj", # deepseek2
+            "language_model.model.layers.{bid}.self_attn.q_a_proj", # deepseek2
         ),
 
         MODEL_TENSOR.ATTN_Q_B: (
-            "model.layers.{bid}.self_attn.q_b_proj", # deepseek2
+            "language_model.model.layers.{bid}.self_attn.q_b_proj", # deepseek2
         ),
 
         MODEL_TENSOR.ATTN_KV_A_MQA: (
-            "model.layers.{bid}.self_attn.kv_a_proj_with_mqa", # deepseek2
+            "language_model.model.layers.{bid}.self_attn.kv_a_proj_with_mqa", # deepseek2
         ),
 
         MODEL_TENSOR.ATTN_KV_B: (
-            "model.layers.{bid}.self_attn.kv_b_proj", # deepseek2
+            "language_model.model.layers.{bid}.self_attn.kv_b_proj", # deepseek2
         ),
 
         MODEL_TENSOR.ATTN_Q_A_NORM: (
-            "model.layers.{bid}.self_attn.q_a_layernorm", # deepseek2
+            "language_model.model.layers.{bid}.self_attn.q_a_layernorm", # deepseek2
         ),
 
         MODEL_TENSOR.ATTN_KV_A_NORM: (
-            "model.layers.{bid}.self_attn.kv_a_layernorm", # deepseek2
+            "language_model.model.layers.{bid}.self_attn.kv_a_layernorm", # deepseek2
         ),
     }
 
@@ -419,10 +419,10 @@ class TensorNameMap:
     arch_block_mappings_cfg: dict[MODEL_ARCH, dict[MODEL_TENSOR, tuple[str, ...]]] = {
         MODEL_ARCH.ARCTIC: {
             MODEL_TENSOR.FFN_NORM: (
-                "model.layers.{bid}.residual_layernorm",
+                "language_model.model.layers.{bid}.residual_layernorm",
             ),
             MODEL_TENSOR.FFN_NORM_EXP: (
-                "model.layers.{bid}.post_attention_layernorm",
+                "language_model.model.layers.{bid}.post_attention_layernorm",
             ),
         },
     }
```

</details>

## HF Transformers の疑問点を追う

HF Transformers & Torchの知識が圧倒的に足りない。
そこを補いながらつめていく。

疑問点:

* 層の一覧表示の方法は?
* model.to(0) の意味は?
* 層の間の接続はどのように定義されているのか?
    * 層の名前付けのルールは?
* 量子化(AWQ)の方法は?
* ファインチューンの方法は?
* プリプロセッサの構成は?
    * テキストエンコーダー、デコーダー
    * 画像エンコーダー

### 層の一覧表示の方法

`named_parameters()` がジェネレーターを返す。

```python
for name, param in model.named_parameters():
    print(f"{name:80s} {str(param.shape):30s} {param.dtype}")
```

参照: [./list-layers.ipynb](./list-layers.ipynb) 

### model.to(0) の意味

`to(0)` をすることでモデルがGPUに移されるらしい。
`0` はデバイス番号かなにかか。

実体は [`torch.nn.Moddule.to`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to) だった。

<https://discuss.pytorch.org/t/model-cuda-vs-model-to-device/93343> によれば `model.cuda()` と `model.to(device)` は同じ効果である。

`device` に数字 `0` が指定されているけれども、cudaであろうことは疑いようがない。
今はこれ以上、深入りしなくてよいだろう。

### 層間の接続はどのように定義されているのか

読み込みは `PreTrainedModel.from_pretrained()` で行われる。

transforms/models/llava や paligemma あたりにそれっぽい参照構造がある。
models/ 下にその連結構造を規定するクラスが置いてあるってことで良かろう。

* languange\_model: LlamaForCausalLM
* vision\_tower: SiglipVisionTransformer

### CPUだけでの推論

`{param.is_cuda}` で各層がCUDAに展開されているか分かる。
やってみると全てのレイヤーがCUDAになった。
もしかして…仮想メモリ的にあつかってるかしら?

CPUで推論させたらloadが100%へ張り付かない。
マルチスレッド実装がされてないか甘いと思われる。
1時間ほどその状態が続いた後、100%になる状態が約10分続き推論が完了した。

vision towerの計算は並列でできない、みたいなことがあるのかもしれない。

推論にかかった時間は以下の通り。

```
CPU times: total: 3h 6s
Wall time: 1h 11min 2s
```

ここから以下の式で100%に張り付いていた時間を推定する。
7分ちょっとくらい。感覚的には合致する。

    (3h6s - 1h11min2s) / 15 = 436s = 7m16s

## GPUを用いた学習

<https://huggingface.co/docs/transformers/ja/model_memory_anatomy>

学習時のメモリはモデルサイズよりも大きくなる。
どのくらい大きくなるかはバッチサイズによる。
バッチサイズは大きいほど収束が速かったり、最終的な性能向上が見込めたりする。
例示されたコードでは1.3GBのモデルの学習に7.5GBのメモリが必要になっている。
約5倍。

そこからなぜそのようなメモリが必要なのか、
どこが速度的なボトルネックになるのかを、この記事は解説している。

<https://huggingface.co/docs/transformers/ja/perf_train_gpu_one>

Trainerを使う方法とPyTorchを使う方法がある。

## 未分類

### 失敗: Flash Attention 2

Flash Attention 2というのを使うと推論が速くなる場合があるらしい。
ネイティブモジュールのコンパイルが必要でそれに失敗するためNG

    pip install -U flash-attn --no-build-isolation

### 失敗: DeepSpeed

DeepSpeedというのを使うと学習が速くなるというのでインストールしようとしたが失敗。
プレコンパイルの async\_io が別途必要らしい。
そして async\_io はWindowsに未対応。
DeepSpeedはmicrosoft製…
