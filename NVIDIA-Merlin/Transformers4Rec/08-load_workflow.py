#!/usr/bin/python3

import os
from pprint import pp

from nvtabular.workflow import Workflow

INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "/workspace/data")

workflow = Workflow.load(os.path.join(INPUT_DATA_DIR, "workflow_etl"))

#pprint.pp(workflow.input_dtypes)
#pprint.pp(workflow.output_dtypes)


from merlin.io import Dataset

ds = Dataset(os.path.join(INPUT_DATA_DIR, "Oct-2019.parquet"))
input = Dataset(ds.head(17))
pp(input.tail())


output = workflow.transform(input)
pp(output.head())
