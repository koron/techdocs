#!/usr/bin/python
#
# 全入力を推論して、結果一覧を得る
#
# USAGE: ./13-inference.py -d {INPUT TSV} [OPTIONS]
#
# OPTIONS:
#
#   -d/--data       入力となるTSVデータファイル (要素はID, expected, image URL, category, image relative pathの5つ)
#   -i/--imageroot  画像の√ディレクトリ (デフォルト: データファイルのディレクトリ)
#   -b/--batchsize  推論のバッチサイズ (デフォルト: 4)

PROMPT = "Answer yes or no to whether this image is approved as an ad by Google Adsense." # 19 tokens

TOKENIZER_PATH = "/root/.cache/paligemma_tokenizer.model"

#MODEL_PATH = "/root/.cache/kagglehub/models/google/paligemma/jax/paligemma-3b-pt-224/1/paligemma-3b-pt-224.f16.npz"
#LLM_VARIANT = "gemma_2b"

MODEL_PATH = "/root/.cache/kagglehub/models/google/paligemma-2/jax/paligemma2-3b-pt-224/1/paligemma2-3b-pt-224.b16.npz"
LLM_VARIANT = "gemma2_2b"

SEQLEN = 128
SAMPLER = "greedy"

IMAGE_ROOT = ""

from PIL import Image
import io
import os.path
import functools
import logging

import numpy as np
import jax, jax.extend
import jax.numpy as jnp
import ml_collections
from big_vision.models.proj.paligemma import paligemma
from big_vision.trainers.proj.paligemma import predict_fns
import big_vision.utils
import big_vision.sharding
import big_vision.datasets.jsonl
import sentencepiece
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

##############################################################################
# Initialize model, parameters, and tokenizer.

g_tokenizer = None
g_model = None
g_params = None

mesh = jax.sharding.Mesh(jax.devices(), ("data"))
data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))

def init_model_and_tokenizer(model_path, tokenizer_path):
    global g_model
    global g_params
    global g_tokenizer
    global g_decode_fn
    global g_decode
    g_tokenizer = sentencepiece.SentencePieceProcessor(TOKENIZER_PATH)
    g_model, g_params = load_model_and_params(model_path)
    g_decode_fn = predict_fns.get_all(g_model)['decode']
    g_decode = functools.partial(g_decode_fn, devices=jax.devices(), eos_token=g_tokenizer.eos_id())

def load_model_and_params(model_path):
    model_config = ml_collections.FrozenConfigDict({
        "llm": {"vocab_size": 257_152, "variant": LLM_VARIANT, "final_logits_softcap": 0.0},
        "img": {"variant": "So400m/14", "pool_type": "none", "scan": True, "dtype_mm": "float16"}
    })
    model = paligemma.Model(**model_config)
    params = paligemma.load(None, model_path, model_config)
    return model, params

##############################################################################
# Inference

def inference_one(text, image_name):
    item = compose_item(text, image_name)
    item["_mask"] = np.array(True)
    batch = jax.tree.map(lambda *x: np.stack(x), *[item])
    batch = big_vision.utils.reshard(batch, data_sharding)
    tokens = g_decode({"params": g_params}, batch=batch, max_decode_len=SEQLEN, sampler=SAMPLER)
    tokens, mask = jax.device_get((tokens, batch["_mask"]))
    tokens = tokens[mask]  # remove padding examples.
    responses = [postprocess_tokens(t) for t in tokens]
    return responses[0]

def inference_batch(text, images):
    items = [compose_item(text, i) for i in images]
    for i in range(len(items)):
        items[i]["_mask"] = np.array(True)
    batch = jax.tree.map(lambda *x: np.stack(x), *items)
    batch = big_vision.utils.reshard(batch, data_sharding)
    tokens = g_decode({"params": g_params}, batch=batch, max_decode_len=SEQLEN, sampler=SAMPLER)
    tokens, mask = jax.device_get((tokens, batch["_mask"]))
    tokens = tokens[mask]  # remove padding examples.
    responses = [postprocess_tokens(t) for t in tokens]
    return responses

def compose_item(prefix, image_name, suffix=None):
    image = load_image(image_name)
    tokens, mask_ar, _, mask_input = preprocess_tokens(prefix, suffix=suffix, seqlen=SEQLEN)
    item = {
        "image": np.asarray(image),
        "text": np.asarray(tokens),
        "mask_ar": np.asarray(mask_ar),
        "mask_input": np.asarray(mask_input),
    }
    return item

def load_image(image_name):
    image_path = os.path.join(IMAGE_ROOT, image_name)
    image = Image.open(image_path)
    image = preprocess_image(image)
    return image

def preprocess_image(image, size=224):
    # Model has been trained to handle images of different aspects ratios
    # resized to 224x224 in the range [-1, 1]. Bilinear and antialias resize
    # options are helpful to improve quality in some tasks.
    image = np.asarray(image)
    if image.ndim == 2:
        # Convert image without last channel into greyscale.
        image = np.stack((image,)*3, axis=-1)
    image = image[..., :3]  # Remove alpha layer.
    assert image.shape[-1] == 3
    #image = tf.constant(image)
    #image = tf.image.resize(image, (size, size), method='bilinear', antialias=True)
    #image = image.numpy()
    return image / 127.5 - 1.0  # [0, 255]->[-1,1]

def preprocess_tokens(prefix, suffix=None, seqlen=None):
    # Model has been trained to handle tokenized text composed of a prefix with
    # full attention and a suffix with causal attention.
    separator = "\n"
    tokens = g_tokenizer.encode(prefix, add_bos=True) + g_tokenizer.encode(separator)
    mask_ar = [0] * len(tokens)    # 0 to use full attention for prefix.
    mask_loss = [0] * len(tokens)  # 0 to not use prefix tokens in the loss.
    if suffix:
        suffix = g_tokenizer.encode(suffix, add_eos=True)
        tokens += suffix
        mask_ar += [1] * len(suffix)    # 1 to use causal attention for suffix.
        mask_loss += [1] * len(suffix)  # 1 to use suffix tokens in the loss.
    mask_input = [1] * len(tokens)    # 1 if its a token, 0 if padding.
    if seqlen:
        padding = [0] * max(0, seqlen - len(tokens))
        tokens = tokens[:seqlen] + padding
        mask_ar = mask_ar[:seqlen] + padding
        mask_loss = mask_loss[:seqlen] + padding
        mask_input = mask_input[:seqlen] + padding
    return jax.tree.map(np.array, (tokens, mask_ar, mask_loss, mask_input))

def postprocess_tokens(tokens):
    tokens = tokens.tolist()  # np.array to list[int]
    try: # Remove tokens at and after EOS if any.
        eos_pos = tokens.index(g_tokenizer.eos_id())
        tokens = tokens[:eos_pos]
    except ValueError:
        pass
    return g_tokenizer.decode(tokens)

##############################################################################
# main

def str_to_bool(s):
    if s.lower() == 'true':
        return True
    else:
        return False

def niter(iter, n):
    alive = True
    while alive:
        chunk = []
        for _ in range(n):
            try:
                chunk.append(next(iter))
            except StopIteration:
                alive = False
        if len(chunk) > 0:
            yield chunk

if __name__ == '__main__':
    import sys
    import argparse
    import csv
    import datetime

    data_path = None
    batch_size = 4

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help='data samples (TSV)')
    parser.add_argument("-i", "--imageroot", help='image root directory')
    parser.add_argument("-b", "--batchsize", help='batch size')
    args, remains = parser.parse_known_args(sys.argv[1:])
    for key, value in vars(args).items():
        match key:
            case "data":
                if value is not None:
                    data_path = value
            case "imageroot":
                if value is not None:
                    IMAGE_ROOT = value
            case "batchsize":
                if value is not None:
                    batch_size = int(value)

    if data_path is None:
        raise RuntimeError("-d/--data is required")
    data_dir = os.path.dirname(data_path)
    if IMAGE_ROOT == "":
        IMAGE_ROOT = data_dir

    init_model_and_tokenizer(MODEL_PATH, TOKENIZER_PATH)

    with open(data_path) as f:
        n = 0
        print(f"#\t{datetime.datetime.now()}\t{n}", file=sys.stderr)
        for rows in niter(csv.reader(f, delimiter="\t"), batch_size):
            images = [row[4] for row in rows]
            results = inference_batch(PROMPT, images)
            for i in range(batch_size):
                id = rows[i][0]
                approve = str_to_bool(rows[i][1])
                category = rows[i][3]
                image_name = rows[i][4]
                result = results[i].replace("\n", "\\n")
                print(f"{id}\t{approve}\t{category}\t{image_name}\t{result}", flush=True)
            n += batch_size
            print(f"#\t{datetime.datetime.now()}\t{n}", file=sys.stderr)

    sys.exit(0)
