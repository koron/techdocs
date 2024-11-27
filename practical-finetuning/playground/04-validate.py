#!/usr/bin/python
#
# 1つのモデルに対して複数データセットを一度にvalidationできるスクリプト
#
# USAGE: ./04-validate.y -m {trained-checkpoint} [JSONL files...]

PROMPT = "Is this image appropriate based on Federal Food, Drug, and Cosmetic Act?"
MODEL_PATH = "/root/.cache/kagglehub/models/google/paligemma/jax/paligemma-3b-pt-224/1/paligemma-3b-pt-224.f16.npz"
TOKENIZER_PATH = "/root/.cache/paligemma_tokenizer.model"

SEQLEN = 128
SAMPLER = "greedy"

DATA_DIR = "dataset"
IMAGE_ROOT = DATA_DIR

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

g_tokenizer = None
g_model = None
g_params = None

mesh = jax.sharding.Mesh(jax.devices(), ("data"))
data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))

def load_model_and_params(model_path):
    model_config = ml_collections.FrozenConfigDict({
        "llm": {"vocab_size": 257_152},
        "img": {"variant": "So400m/14", "pool_type": "none", "scan": True, "dtype_mm": "float16"}
    })
    model = paligemma.Model(**model_config)
    params = paligemma.load(None, model_path, model_config)
    return model, params

def shard_params(params):
    params_sharding = big_vision.sharding.infer_sharding(params, strategy=[('.*', 'fsdp(axis="data")')], mesh=mesh)
    params, treedef = jax.tree.flatten(params)
    sharding_leaves = jax.tree.leaves(params_sharding)
    for idx, sharding in enumerate(sharding_leaves):
        params[idx] = big_vision.utils.reshard(params[idx], sharding)
        params[idx].block_until_ready()
    params = jax.tree.unflatten(treedef, params)
    return params

def load_tokenizer(path):
    return sentencepiece.SentencePieceProcessor(path)

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
    
    image = tf.constant(image)
    image = tf.image.resize(image, (size, size), method='bilinear', antialias=True)
    return image.numpy() / 127.5 - 1.0  # [0, 255]->[-1,1]

def preprocess_tokens(prefix, suffix=None, seqlen=None):
    global g_tokenizer
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
    global g_tokenizer
    tokens = tokens.tolist()  # np.array to list[int]
    try:  # Remove tokens at and after EOS if any.
        eos_pos = tokens.index(g_tokenizer.eos_id())
        tokens = tokens[:eos_pos]
    except ValueError:
        pass
    return g_tokenizer.decode(tokens)

def data_iterator(data_path):
    datadir = IMAGE_ROOT.encode()
    dataset = big_vision.datasets.jsonl.DataSource(data_path)
    for item in dataset.get_tfdata(ordered=True).as_numpy_iterator():
        imgpath = os.path.join(datadir, item["image"])
        image = Image.open(imgpath)
        image = preprocess_image(image)
        prefix = PROMPT
        suffix = item["suffix"].decode().lower()
        tokens, mask_ar, _, mask_input = preprocess_tokens(prefix, seqlen=SEQLEN)
        yield {
                "item": {
                    "image": item["image"].decode().lower(),
                    "suffix": suffix,
                },
                "example": {
                    "image": np.asarray(image),
                    "text": np.asarray(tokens),
                    "mask_ar": np.asarray(mask_ar),
                    "mask_input": np.asarray(mask_input),
                },
        }

def inference(decode, examples):
    global g_params
    for i in range(len(examples)):
        examples[i]["_mask"] = np.array(True)
    batch = jax.tree.map(lambda *x: np.stack(x), *examples)
    batch = big_vision.utils.reshard(batch, data_sharding)
    tokens = decode({"params": g_params}, batch=batch, max_decode_len=SEQLEN, sampler=SAMPLER)
    tokens, mask = jax.device_get((tokens, batch["_mask"]))
    tokens = tokens[mask]  # remove padding examples.
    responses = [postprocess_tokens(t) for t in tokens]
    return responses

def model_validate(model_path, validations):
    global g_model
    global g_params
    global g_tokenizer
    g_tokenizer = load_tokenizer(TOKENIZER_PATH)
    logging.info("loading")
    g_model, g_params = load_model_and_params(model_path)
    logging.info("sharding")
    g_params = shard_params(g_params)
    decode_fn = predict_fns.get_all(g_model)['decode']
    decode = functools.partial(decode_fn, devices=jax.devices(), eos_token=g_tokenizer.eos_id())
    for v in validations:
        logging.info(f"inferencing: {v}")
        Tp = 0; Tn = 0; Fp = 0; Fn = 0
        for item in data_iterator(v):
            responses = inference(decode, [item["example"]])
            want = item["item"]["suffix"]
            got = responses[0]
            if want == got:
                if want == "yes":
                    Tp += 1
                else:
                    Tn += 1
            else:
                if want == "yes":
                    Fn += 1
                else:
                    Fp += 1
            logging.info(f"{item['item']['image']} - want={want} got={got}")
        logging.info("complete")
        print(f"Result: {v}")
        print(f"  Tp={Tp} Tn={Tn} Fp={Fp} Fn={Fn}")
        accuracy = (Tp + Tn) / (Tp + Tn + Fp + Fn)
        print(f"  accuracy:  {accuracy}")
        precision = 0
        if Tp + Fp > 0:
            precision = Tp / (Tp + Fp)
            print(f"  precision: {precision}")
        if Tp + Fn > 0:
            recall = Tp / (Tp + Fn)
            print(f"  recall:    {recall}")
        if precision != 0 and recall != 0:
            fmeasure = 2 * precision * recall / (precision + recall)
            print(f"  F-measure: {fmeasure}")

if __name__ == '__main__':
    from sys import argv
    import argparse

    model = MODEL_PATH
    validations = []

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help='model to input')
    args, remains = parser.parse_known_args(argv[1:])
    for key, value in vars(args).items():
        match key:
            case "model":
                if value is not None:
                    model = value

    logging.basicConfig(level=logging.WARN, format='%(asctime)s %(message)s')

    if len(remains) > 0:
        validations = remains

    model_validate(model, validations)
