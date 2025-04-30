#!/usr/bin/python3


import os

import numpy as np
import pandas as pd

from pprint import pp


# compose dataframe

cols = [
        #'user_session',
        #'product_id-count',
        'product_id-list',
        #'category_code-list',
        'brand-list',
        'category_id-list',
        'et_dayofweek_sin-list',
        'et_dayofweek_cos-list',
        'price_log_norm-list',
        'relative_price_to_avg_categ_id-list',
        'product_recency_days_log_norm-list',
        #'day_index',
        ]

emp = np.empty(0, dtype=np.int64)

data = [
    [ emp   , emp, emp, emp, emp, emp, emp, emp, ], # 何も買ってない(初めての買い物客)
    [ [   1], [0], [0], [0], [1], [0], [0], [0], ], # id:1 の商品を0°曜日に買った
    [ [   1], [0], [0], [1], [0], [0], [0], [0], ], # id:1 の商品を90°曜日に買った
    [ [9999], [0], [0], [0], [1], [0], [0], [0], ], # id:9999 の商品を0°曜日に買った

    #[ 999999999, 1, [60], [0], [0], [0], [0], [0], [0], [0], [0], 1, ],
    #[ 999999999, 1, [90], [0], [0], [0], [0], [0], [0], [0], [0], 1, ],
]

df = pd.DataFrame(data, columns=cols)
pp(df)


# load pre-trained model

import cloudpickle

INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "/workspace/data")

model = cloudpickle.load(open(os.path.join(INPUT_DATA_DIR, "trained_model1", "t4rec_model_class.pkl"), "rb"))


# setup the trainer

from merlin.schema import Schema
from merlin.io import Dataset

# Define categorical and continuous columns to fed to training model
x_cat_names = ['product_id-list', 'category_id-list', 'brand-list']
x_cont_names = ['product_recency_days_log_norm-list', 'et_dayofweek_sin-list', 'et_dayofweek_cos-list',
                'price_log_norm-list', 'relative_price_to_avg_categ_id-list']

train = Dataset(os.path.join(INPUT_DATA_DIR, "processed_nvt/part_0.parquet"))
schema = train.schema
schema = schema.select_by_name(x_cat_names + x_cont_names)


from transformers4rec.config.trainer import T4RecTrainingArguments
from transformers4rec.torch import Trainer

#Set arguments for training
training_args = T4RecTrainingArguments(
            output_dir = "./tmp",
            max_sequence_length=20,
            data_loader_engine='merlin',
            num_train_epochs=3,
            dataloader_drop_last=False,
            per_device_train_batch_size = 256,
            per_device_eval_batch_size = 32,
            gradient_accumulation_steps = 1,
            learning_rate=0.000666,
            report_to = [],
            logging_steps=200,
        )


# Instantiate the T4Rec Trainer, which manages training and evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    schema=schema,
    compute_metrics=True,
)


# predict

import torch

model.eval()
with torch.no_grad():
    ds = Dataset(df)
    out = trainer.predict(ds)
    items = out.predictions[0]
    print(items)
    #logits = out.predictions[1]
    #for i in range(len(items)):
    #    print(f"  item#{i}   {items[i]}")
    #    print(f"  logits#{i} {logits[i]}")
    #    print()
