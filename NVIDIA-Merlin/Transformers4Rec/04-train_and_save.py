#!/usr/bin/python3


import os
import glob

import torch
import transformers4rec.torch as tr

from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt
from transformers4rec.torch.utils.examples_utils import wipe_memory


from merlin.schema import Schema
from merlin.io import Dataset

# Define categorical and continuous columns to fed to training model
x_cat_names = ['product_id-list', 'category_id-list', 'brand-list']
x_cont_names = ['product_recency_days_log_norm-list', 'et_dayofweek_sin-list', 'et_dayofweek_cos-list',
                'price_log_norm-list', 'relative_price_to_avg_categ_id-list']


INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "/workspace/data")

train = Dataset(os.path.join(INPUT_DATA_DIR, "processed_nvt/part_0.parquet"))
schema = train.schema
schema = schema.select_by_name(x_cat_names + x_cont_names)


# Define input block
sequence_length, d_model = 20, 192
# Define input module to process tabular input-features and to prepare masked inputs
inputs= tr.TabularSequenceFeatures.from_schema(
    schema,
    max_sequence_length=sequence_length,
    d_output=d_model,
    masking="mlm",
    aggregation="concat",
)


# Define XLNetConfig class and set default parameters for HF XLNet config
transformer_config = tr.XLNetConfig.build(
    d_model=d_model, n_head=4, n_layer=2, total_seq_length=sequence_length
)
# Define the model block including: inputs, masking, projection and transformer block.
body = tr.SequentialBlock(
    inputs, tr.MLPBlock([192]), tr.TransformerBlock(transformer_config, masking=inputs.masking)
)

# Define the head related to next item prediction task
head = tr.Head(
    body,
    tr.NextItemPredictionTask(weight_tying=True,
                              metrics=[NDCGAt(top_ks=[10, 20], labels_onehot=True),
                                       RecallAt(top_ks=[10, 20], labels_onehot=True)]),
)

# Get the end-to-end Model class
model = tr.Model(head)


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


OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/data/sessions_by_day")


#%%time
start_time_window_index = 1
final_time_window_index = 4
for time_index in range(start_time_window_index, final_time_window_index):
    # Set data
    time_index_train = time_index
    time_index_eval = time_index + 1
    train_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_train}/train.parquet"))
    eval_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_eval}/valid.parquet"))

    # Initialize dataloaders
    trainer.train_dataset_or_path = train_paths
    trainer.eval_dataset_or_path = eval_paths

    # Train on day related to time_index
    print('*'*20)
    print("Launch training for day %s are:" %time_index)
    print('*'*20 + '\n')
    trainer.reset_lr_scheduler()
    trainer.train()
    trainer.state.global_step +=1

    # Evaluate on the following day
    train_metrics = trainer.evaluate(metric_key_prefix='eval')
    print('*'*20)
    print("Eval results for day %s are:\t" %time_index_eval)
    print('\n' + '*'*20 + '\n')
    for key in sorted(train_metrics.keys()):
        print(" %s = %s" % (key, str(train_metrics[key])))
    wipe_memory()

model.save(os.path.join(INPUT_DATA_DIR, "trained_model1"))
#trainer.save_model(os.path.join(INPUT_DATA_DIR, "trained_model2"))
