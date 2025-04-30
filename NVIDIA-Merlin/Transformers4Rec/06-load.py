#!/usr/bin/python3

import os
import cloudpickle

INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "/workspace/data")

model = cloudpickle.load(
        open(os.path.join(INPUT_DATA_DIR, "trained_model1", "t4rec_model_class.pkl"), "rb")
        )
