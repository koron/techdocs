#!/usr/bin/python3

import os
import cloudpickle

INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "/workspace/data")

model = cloudpickle.load(open(os.path.join(INPUT_DATA_DIR, "trained_model1", "t4rec_model_class.pkl"), "rb"))


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


import torch
import glob

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/data/sessions_by_day")

# Set the model to evaluation mode.
model.eval()
with torch.no_grad():
    for i in range(1, 8):
        eval_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{i}/valid.parquet"))
        trainer.eval_dataset_or_path = eval_paths

        # Evaluate on the following day
        train_metrics = trainer.evaluate(metric_key_prefix='eval')
        print('*'*20)
        print("Eval results for day %s are:\t" %i)
        print('\n' + '*'*20 + '\n')
        for key in sorted(train_metrics.keys()):
            print(" %s = %s" % (key, str(train_metrics[key])))
