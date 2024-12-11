#!/usr/bin/python
#
# Script to tokenize STDIN
#
# 標準入力の内容をトークナイザーでトークン列に変換し、カウントする

TOKENIZER_PATH = "/root/.cache/paligemma_tokenizer.model"

import sys
import sentencepiece

tokenizer = sentencepiece.SentencePieceProcessor(TOKENIZER_PATH)
text = sys.stdin.read()
tokens = tokenizer.encode(text)
print(tokens)
print("len=", len(tokens))
