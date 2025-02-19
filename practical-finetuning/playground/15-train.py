#!/usr/bin/python
#

PROMPT = "Answer yes or no to whether this image is approved as an ad by Google Adsense." # 19 tokens

TOKENIZER_PATH = "/root/.cache/paligemma_tokenizer.model"

#MODEL_PATH = "/root/.cache/kagglehub/models/google/paligemma/jax/paligemma-3b-pt-224/1/paligemma-3b-pt-224.f16.npz"
#LLM_VARIANT = "gemma_2b"

MODEL_PATH = "/root/.cache/kagglehub/models/google/paligemma-2/jax/paligemma2-3b-pt-224/1/paligemma2-3b-pt-224.b16.npz"
LLM_VARIANT = "gemma2_2b"

SEQLEN = 128
SAMPLER = "greedy"

IMAGE_ROOT = ""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

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
#import tensorflow as tf

#tf.config.set_visible_devices([], "GPU")

##############################################################################
# Initialize model, parameters, and tokenizer.

g_tokenizer = None
g_model = None
g_params = None
g_trainable_mask = None

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

def get_trainable_mask(params):
    def is_trainable_param(name, param):
        if name.startswith("llm/layers/attn/"):
            return True
        if name.startswith("llm/"):
            return False
        if name.startswith("img/"):
            return False
        raise ValueError(f"Unexpected param name {name}")
    return big_vision.utils.tree_map_with_names(is_trainable_param, params)

##############################################################################
# Inference

def inference_batch(text, images):
    batch = create_inference_batch(text, images)
    tokens = g_decode({"params": g_params}, batch=batch, max_decode_len=SEQLEN, sampler=SAMPLER)
    tokens, mask = jax.device_get((tokens, batch["_mask"]))
    tokens = tokens[mask]  # remove padding examples.
    responses = [postprocess_tokens(t) for t in tokens]
    return responses

def create_inference_batch(text, images):
    items = [compose_inference_item(text, i) for i in images]
    for i in range(len(items)):
        items[i]["_mask"] = np.array(True)
    batch = jax.tree.map(lambda *x: np.stack(x), *items)
    batch = big_vision.utils.reshard(batch, data_sharding)
    return batch

def compose_inference_item(prefix, image_name):
    image = load_image(image_name)
    tokens, mask_ar, _, mask_input = preprocess_tokens(prefix, seqlen=SEQLEN)
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
# Training

def create_training_batch(text, images, wants):
    items = [compose_training_item(text, images[i], wants[i]) for i in range(0, len(images))]
    batch = jax.tree.map(lambda *x: np.stack(x), *items)
    batch = big_vision.utils.reshard(batch, data_sharding)
    return batch

def compose_training_item(prefix, image_name, suffix):
    #print(f"image_name={image_name} suffix={suffix}")
    image = load_image(image_name)
    tokens, mask_ar, mask_loss, _ = preprocess_tokens(prefix, suffix=suffix, seqlen=SEQLEN)
    item = {
        "image": np.asarray(image),
        "text": np.asarray(tokens),
        "mask_ar": np.asarray(mask_ar),
        "mask_loss": np.asarray(mask_loss),
    }
    return item

@functools.partial(jax.jit, donate_argnums=(0,))
def update_fn(params, batch, learning_rate):
  imgs, txts, mask_ar = batch["image"], batch["text"], batch["mask_ar"]

  def loss_fn(params):
    text_logits, _ = g_model.apply({"params": params}, imgs, txts[:, :-1], mask_ar[:, :-1], train=True)
    logp = jax.nn.log_softmax(text_logits, axis=-1)

    # The model takes as input txts[:, :-1] but the loss is defined as predicting
    # next tokens txts[:, 1:]. Additionally, mask_loss[:, 1:] indicates which tokens
    # are part of the loss (e.g. prefix and padded tokens are not included).
    mask_loss = batch["mask_loss"][:, 1:]
    targets = jax.nn.one_hot(txts[:, 1:], text_logits.shape[-1])

    # Compute the loss per example. i.e. the mean of per token pplx.
    # Since each example has a different number of tokens we normalize it.
    token_pplx = jnp.sum(logp * targets, axis=-1)  # sum across vocab_size.
    example_loss = -jnp.sum(token_pplx * mask_loss, axis=-1)  # sum across seq_len.
    example_loss /= jnp.clip(jnp.sum(mask_loss, -1), 1)  # weight by num of tokens.

    # batch_loss: mean of per example loss.
    return jnp.mean(example_loss)

  loss, grads = jax.value_and_grad(loss_fn)(params)

  # Apply gradients to trainable params using SGD.
  def apply_grad(param, gradient, trainable):
    if not trainable:
        return param
    return param - learning_rate * gradient

  params = jax.tree_util.tree_map(apply_grad, params, grads, g_trainable_mask)

  return params, loss

##############################################################################
# Utilities for main

import random

def str_to_bool(s):
    s = s.lower()
    if s == 'true' or s == 'yes':
        return True
    else:
        return False

def to_want(s):
    if str_to_bool(s):
        return 'yes'
    else:
        return 'no'

def to_got(s):
    if str_to_bool(s):
        return 'yes'
    else:
        return 'no'

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

def read_csv_all(name):
    with open(name) as f:
        r = csv.reader(f, delimiter="\t")
        return [item for item in r]

def shuffle_repeat_iter(all):
    while True:
        batch = random.sample(all, len(all))
        for item in batch:
            yield item

def repeat_iter(items, n):
    for i in range(0, n):
        batch = random.sample(items, len(items))
        for item in batch:
            yield item

##############################################################################
# main

if __name__ == '__main__':
    import sys
    import argparse
    import csv
    import datetime

    data_path = None
    batch_size = 4
    train_iteration = 1
    learning_rate = 0.03
    save_params = ''

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", help='load a file as model parameters')
    parser.add_argument("-d", "--data", help='data for training (TSV)')
    parser.add_argument("-i", "--imageroot", help='image root directory')
    parser.add_argument("-b", "--batchsize", help='batch size')
    parser.add_argument("-t", "--trainiteration", help='training iteration')
    parser.add_argument("-r", "--learningrate", help='learning rate')
    parser.add_argument("-s", "--save", help='save parameters after training')
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
            case "trainiteration":
                if value is not None:
                    train_iteration = int(value)
            case "learningrate":
                if value is not None:
                    learning_rate = float(value)
            case "save":
                if value is not None:
                    save_params = value
            case "load":
                if value is not None:
                    MODEL_PATH = value

    if data_path is None:
        raise RuntimeError("-d/--data is required")
    data_dir = os.path.dirname(data_path)
    if IMAGE_ROOT == "":
        IMAGE_ROOT = data_dir

    train_examples = read_csv_all(data_path)
    train_steps = (len(train_examples) * train_iteration + batch_size - 1) // batch_size
    train_iter = repeat_iter(train_examples, train_iteration)

    #print(f"len(train_examples)={len(train_examples)}")
    #print(f"train_steps={train_steps}")
    #print(f"learning_rate={learning_rate}")

    init_model_and_tokenizer(MODEL_PATH, TOKENIZER_PATH)
    g_trainable_mask = get_trainable_mask(g_params)

    sched_fn = big_vision.utils.create_learning_rate_schedule(
        total_steps=train_steps+1, base=learning_rate,
        decay_type="cosine", warmup_percent=0.10)

    step = 0
    for rows in niter(train_iter, batch_size):
        step += 1
        #for row in rows:
        #    print(f"step:{step} id:{row[0]} want:{to_want(row[1])} got:{to_got(row[5])}")
        images = [row[4] for row in rows]
        wants = [to_want(row[1]) for row in rows]
        batch = create_training_batch(PROMPT, images, wants)
        rate = sched_fn(step)
        g_params, loss = update_fn(g_params, batch, rate)
        loss = jax.device_get(loss)
        if True:
            print(f"  {datetime.datetime.now()}  step:{step:2d}/{train_steps:2d}  lr:{learning_rate:.5f}  loss: {loss:.4f}")

    if save_params != "":
        print(f"saving as {save_params}")
        with open(save_params, "wb") as f:
            flat, _ = big_vision.utils.tree_flatten_with_names(g_params)
            np.savez(f, **{k: v for k, v in flat})

    sys.exit(0)
