#!/usr/bin/python
#
# Script to download models for paligemma2 and tokenizer.
#
# PaliGemma2自身とトークナイザーのモデルをダウンロードする
# ダウンロード先は /root/.cache/ ディレクトリ

import os

MODEL_PATH = "./paligemma2-3b-pt-224.b16.npz"

if not os.path.exists(MODEL_PATH):
    import kagglehub
    MODEL_PATH = kagglehub.model_download('google/paligemma-2/jax/paligemma2-3b-pt-224', 'paligemma2-3b-pt-224.b16.npz')

print(f"Model path: {MODEL_PATH}")

TOKENIZER_PATH = "/root/.cache/paligemma_tokenizer.model"

if not os.path.exists(TOKENIZER_PATH):
    import subprocess
    subprocess.run(["gsutil", "cp", "gs://big_vision/paligemma_tokenizer.model", TOKENIZER_PATH])

print(f"Tokenizer path: {TOKENIZER_PATH}")
