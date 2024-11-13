#!/usr/bin/python
#
# 学習してパラメーターを保存するスクリプト
#
# USAGE: ./03-do_learn.py -d {dataset name} -w {model name to write}
#
# Options:
#
# -d    入力に使うデータセット名を指定する。
#       データセット名は ./dataset/{セット名}/ に展開され
#       そこから train.jsonl と valid.jsonl が利用される
#
# -w    学習後のモデルを保存する場合に指定する
#       ./checkpoints/{保存名}.npz で学習後のモデルが保存される
#
# -2    1回目の学習に続けてて2回目の学習を実行する場合に指定する
#       学習率は1回目よりも下げて設定される
#
# -m    学習前のモデルを指定する
#       デフォルトは /root/.cache 下にある事前学習済みのもの
#       ./checkpoints/trained-20241111.npz のようにfinetune済みのものを指定し
#       更なる追加学習を行える

from sys import argv
import os
import functools
import warnings

PROMPT = "Is this image appropriate based on Federal Food, Drug, and Cosmetic Act?"
MODEL_PATH = "/root/.cache/kagglehub/models/google/paligemma/jax/paligemma-3b-pt-224/1/paligemma-3b-pt-224.f16.npz"
TOKENIZER_PATH = "/root/.cache/paligemma_tokenizer.model"


DATA_DIR = "dataset"
DATA_SET = "set1"

IMAGE_ROOT = DATA_DIR

LEARNING_RATE = 0.03
BATCH_SIZE = 2
TRAIN_EXAMPLES = 512

SECOND_TRAINING = False

SAVE_CHECKPOINT = False
MODEL_OUTPUT = "./checkpoints/out.npz"

SEQLEN = 128


# Configure by options

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help='model to input')
parser.add_argument("-d", "--dataset", help='dataset name (default: set1)')
parser.add_argument("-p", "--prompt", help='override prompt')
parser.add_argument("-w", "--write-model", help='write model after training with this name in checkpoints (w/o .npz')
parser.add_argument("-2", "--second-training", action='store_true', help='do second training')

for key, value in vars(parser.parse_args(argv[1:])).items():
    match key:
        case "model":
            if value is not None:
                MODEL_PATH = value
        case "dataset":
            if value is not None:
                DATA_SET = value
        case "prompt":
            if value is not None:
                PROMPT = prompt
        case "write_model":
            if value is not None:
                MODEL_OUTPUT = os.path.join(".", "checkpoints", value + ".npz")
                SAVE_CHECKPOINT = True
        case "second_training":
            if value is True:
                SECOND_TRAINING = True

# Determine parameters

DATA_TRAINNG = os.path.join(DATA_DIR, DATA_SET, "train.jsonl")
DATA_VALIDATION = os.path.join(DATA_DIR, DATA_SET, "valid.jsonl")
DATA_WHOLE = os.path.join(DATA_DIR, "data.jsonl")

# Load tokenizer
import sentencepiece
tokenizer = sentencepiece.SentencePieceProcessor(TOKENIZER_PATH)


# Setup Tensorflow and JAX
import jax, jax.extend
import jax.numpy as jnp
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
backend = jax.extend.backend.get_backend()
print(f"JAX version:  {jax.__version__}")
print(f"JAX platform: {backend.platform}")
print(f"JAX devices:  {jax.device_count()}")


# Define model and load
import ml_collections
from big_vision.models.proj.paligemma import paligemma
model_config = ml_collections.FrozenConfigDict({
    "llm": {"vocab_size": 257_152},
    "img": {"variant": "So400m/14", "pool_type": "none", "scan": True, "dtype_mm": "float16"}
})
model = paligemma.Model(**model_config)
params = paligemma.load(None, MODEL_PATH, model_config)


import big_vision.sharding

def is_trainable_param(name, param):
  if name.startswith("llm/layers/attn/"):  return True
  if name.startswith("llm/"):              return False
  if name.startswith("img/"):              return False
  raise ValueError(f"Unexpected param name {name}")
trainable_mask = big_vision.utils.tree_map_with_names(is_trainable_param, params)

# If more than one device is available (e.g. multiple GPUs) the parameters can
# be sharded across them to reduce HBM usage per device.
mesh = jax.sharding.Mesh(jax.devices(), ("data"))

params_sharding = big_vision.sharding.infer_sharding(
    params, strategy=[('.*', 'fsdp(axis="data")')], mesh=mesh)

# Yes: Some donated buffers are not usable.
warnings.filterwarnings(
    "ignore", message="Some donated buffers were not usable")

@functools.partial(jax.jit, donate_argnums=(0,), static_argnums=(1,))
def maybe_cast_to_f32(params, trainable):
  return jax.tree.map(lambda p, m: p.astype(jnp.float32) if m else p,
                      params, trainable)

# Loading all params in simultaneous - albeit much faster and more succinct -
# requires more RAM than the T4 colab runtimes have by default (12GB RAM).
# Instead we do it param by param.
params, treedef = jax.tree.flatten(params)
sharding_leaves = jax.tree.leaves(params_sharding)
trainable_leaves = jax.tree.leaves(trainable_mask)
for idx, (sharding, trainable) in enumerate(zip(sharding_leaves, trainable_leaves)):
    #print(f"{idx} {sharding} {trainable}")
    params[idx] = big_vision.utils.reshard(params[idx], sharding)
    #params[idx] = maybe_cast_to_f32(params[idx], trainable)
    params[idx].block_until_ready()
params = jax.tree.unflatten(treedef, params)


#
# Utility functions for inference

import numpy as np
from big_vision.trainers.proj.paligemma import predict_fns

# Define `decode` function to sample outputs from the model.
decode_fn = predict_fns.get_all(model)['decode']
decode = functools.partial(decode_fn, devices=jax.devices(), eos_token=tokenizer.eos_id())

data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))

def preprocess_image(image, size=224):
    # Model has been trained to handle images of different aspects ratios
    # resized to 224x224 in the range [-1, 1]. Bilinear and antialias resize
    # options are helpful to improve quality in some tasks.
    image = np.asarray(image)
    if image.ndim == 2:  # Convert image without last channel into greyscale.
        image = np.stack((image,)*3, axis=-1)
    image = image[..., :3]  # Remove alpha layer.
    assert image.shape[-1] == 3

    image = tf.constant(image)
    image = tf.image.resize(image, (size, size), method='bilinear', antialias=True)
    return image.numpy() / 127.5 - 1.0  # [0, 255]->[-1,1]

def preprocess_tokens(prefix, suffix=None, seqlen=None):
    # Model has been trained to handle tokenized text composed of a prefix with
    # full attention and a suffix with causal attention.
    separator = "\n"
    tokens = tokenizer.encode(prefix, add_bos=True) + tokenizer.encode(separator)
    mask_ar = [0] * len(tokens)    # 0 to use full attention for prefix.
    mask_loss = [0] * len(tokens)  # 0 to not use prefix tokens in the loss.

    if suffix:
        suffix = tokenizer.encode(suffix, add_eos=True)
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
    try:  # Remove tokens at and after EOS if any.
        eos_pos = tokens.index(tokenizer.eos_id())
        tokens = tokens[:eos_pos]
    except ValueError:
        pass
    return tokenizer.decode(tokens)


import os.path
import big_vision.datasets.jsonl

train_dataset = big_vision.datasets.jsonl.DataSource(DATA_TRAINNG, fopen_keys={"image": IMAGE_ROOT})

val_dataset = big_vision.datasets.jsonl.DataSource(DATA_VALIDATION, fopen_keys={"image": IMAGE_ROOT})

import io
from PIL import Image

def train_data_iterator():
  """Never ending iterator over training examples."""
  # Shuffle examples and repeat so one can train for many epochs.
  dataset = train_dataset.get_tfdata().shuffle(1_000).repeat()
  for example in dataset.as_numpy_iterator():
    image = Image.open(io.BytesIO(example["image"]))
    image = preprocess_image(image)

    prefix = PROMPT
    suffix = example["suffix"].decode().lower()
    tokens, mask_ar, mask_loss, _ = preprocess_tokens(prefix, suffix, SEQLEN)

    yield {
        "image": np.asarray(image),
        "text": np.asarray(tokens),
        "mask_ar": np.asarray(mask_ar),
        "mask_loss": np.asarray(mask_loss),
    }


def validation_data_iterator():
  """Single iterator over validation examples."""
  for example in val_dataset.get_tfdata(ordered=True).as_numpy_iterator():
    image = Image.open(io.BytesIO(example["image"]))
    image = preprocess_image(image)

    prefix = PROMPT
    tokens, mask_ar, _, mask_input = preprocess_tokens(prefix, seqlen=SEQLEN)

    yield {
        "image": np.asarray(image),
        "text": np.asarray(tokens),
        "mask_ar": np.asarray(mask_ar),
        "mask_input": np.asarray(mask_input),
    }


# @title Define the training step and evaluation loop.
#
# The main update_fn using simple SGD.
#
@functools.partial(jax.jit, donate_argnums=(0,))
def update_fn(params, batch, learning_rate):
  imgs, txts, mask_ar = batch["image"], batch["text"], batch["mask_ar"]

  def loss_fn(params):
    text_logits, _ = model.apply({"params": params}, imgs, txts[:, :-1], mask_ar[:, :-1], train=True)
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
    if not trainable: return param
    return param - learning_rate * gradient

  params = jax.tree_util.tree_map(apply_grad, params, grads, trainable_mask)

  return params, loss


# Validate params.

def data_iterator(data_path):
    image_root = IMAGE_ROOT.encode()
    dataset = big_vision.datasets.jsonl.DataSource(data_path)
    for item in dataset.get_tfdata(ordered=True).as_numpy_iterator():
        imgpath = os.path.join(image_root, item["image"])
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
    global params
    for i in range(len(examples)):
        examples[i]["_mask"] = np.array(True)
    batch = jax.tree.map(lambda *x: np.stack(x), *examples)
    batch = big_vision.utils.reshard(batch, data_sharding)
    tokens = decode({"params": params}, batch=batch, max_decode_len=SEQLEN, sampler="greedy")
    tokens, mask = jax.device_get((tokens, batch["_mask"]))
    tokens = tokens[mask]  # remove padding examples.
    responses = [postprocess_tokens(t) for t in tokens]
    return responses

def validate_model(label):
    for v in [ DATA_WHOLE, DATA_TRAINNG, DATA_VALIDATION]:
        Tp = 0; Tn = 0; Fp = 0; Fn = 0
        for item in data_iterator(v):
            want = item["item"]["suffix"]
            got = inference(decode, [item["example"]])[0]
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
        print(f"Result: {v} ({label})")
        print(f"  Tp={Tp} Tn={Tn} Fp={Fp} Fn={Fn}")
        accuracy = (Tp + Tn) / (Tp + Tn + Fp + Fn)
        print(f"  accuracy:  {accuracy}")
        precision = 0
        if Tp + Fp > 0:
            precision = Tp / (Tp + Fp)
            print(f"  precision: {precision}")
        recall = 0
        if Tp + Fn > 0:
            recall = Tp / (Tp + Fn)
            print(f"  recall:    {recall}")
        if precision != 0 and recall != 0:
            fmeasure = 2 * precision * recall / (precision + recall)
            print(f"  F-measure: {fmeasure}")

validate_model("zero trained")

# Training loop (1st)

TRAIN_STEPS = TRAIN_EXAMPLES // BATCH_SIZE
EVAL_STEPS = TRAIN_STEPS // 8

train_data_it = train_data_iterator()

sched_fn = big_vision.utils.create_learning_rate_schedule(
    total_steps=TRAIN_STEPS+1, base=LEARNING_RATE,
    decay_type="cosine", warmup_percent=0.10)

print(f"1st training: LEARNING_RATE={LEARNING_RATE}")
for step in range(1, TRAIN_STEPS+1):
    # Make list of N training examples.
    examples = [next(train_data_it) for _ in range(BATCH_SIZE)]

    # Convert list of examples into a dict of np.arrays and load onto devices.
    batch = jax.tree.map(lambda *x: np.stack(x), *examples)
    batch = big_vision.utils.reshard(batch, data_sharding)

    # Training step and report training loss
    learning_rate = sched_fn(step)
    params, loss = update_fn(params, batch, learning_rate)

    loss = jax.device_get(loss)
    if (step % EVAL_STEPS) == 0:
        print(f"  step: {step:2d}/{TRAIN_STEPS:2d}   lr: {learning_rate:.5f}   loss: {loss:.4f}")


validate_model("1st trained")

# Training loop (2nd)

if SECOND_TRAINING:
    LEARNING_RATE = 0.003

    train_data_it = train_data_iterator()

    sched_fn = big_vision.utils.create_learning_rate_schedule(
        total_steps=TRAIN_STEPS+1, base=LEARNING_RATE,
        decay_type="cosine", warmup_percent=0.10)

    print(f"2nd training: LEARNING_RATE={LEARNING_RATE}")
    for step in range(1, TRAIN_STEPS+1):
        # Make list of N training examples.
        examples = [next(train_data_it) for _ in range(BATCH_SIZE)]

        # Convert list of examples into a dict of np.arrays and load onto devices.
        batch = jax.tree.map(lambda *x: np.stack(x), *examples)
        batch = big_vision.utils.reshard(batch, data_sharding)

        # Training step and report training loss
        learning_rate = sched_fn(step)
        params, loss = update_fn(params, batch, learning_rate)

        loss = jax.device_get(loss)
        if (step % EVAL_STEPS) == 0:
            print(f"  step: {step:2d}/{TRAIN_STEPS:2d}   lr: {learning_rate:.5f}   loss: {loss:.4f}")

    validate_model("2nd trained")

if SAVE_CHECKPOINT:
    flat, _ = big_vision.utils.tree_flatten_with_names(params)
    with open(MODEL_OUTPUT, "wb") as f:
        np.savez(f, **{k: v for k, v in flat})
