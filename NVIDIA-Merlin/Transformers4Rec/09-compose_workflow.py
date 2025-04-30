#!/usr/bin/python3

# Workflowの構築部分の抜粋:
#
# examples/tutorialからの抜粋なので、
# このファイル単独で何か意味のある動作をするようには作っていない。

import numpy as np
import nvtabular as nvt

from merlin.dag import ColumnSelector
from merlin.schema import Schema, Tags

# categorify features 
item_id = ['product_id'] >> nvt.ops.TagAsItemID()
cat_feats = item_id + ['category_code', 'brand', 'user_id', 'category_id', 'event_type'] >> nvt.ops.Categorify()


# create time features
session_ts = ['event_time_ts']

session_time = (
    session_ts >> 
    nvt.ops.LambdaOp(lambda col: cudf.to_datetime(col, unit='s')) >> 
    nvt.ops.Rename(name = 'event_time_dt')
)

sessiontime_weekday = (
    session_time >> 
    nvt.ops.LambdaOp(lambda col: col.dt.weekday) >> 
    nvt.ops.Rename(name ='et_dayofweek')
)

def get_cycled_feature_value_sin(col, max_value):
    value_scaled = (col + 0.000001) / max_value
    value_sin = np.sin(2*np.pi*value_scaled)
    return value_sin

def get_cycled_feature_value_cos(col, max_value):
    value_scaled = (col + 0.000001) / max_value
    value_cos = np.cos(2*np.pi*value_scaled)
    return value_cos

weekday_sin = (sessiontime_weekday >> 
               (lambda col: get_cycled_feature_value_sin(col+1, 7)) >> 
               nvt.ops.Rename(name = 'et_dayofweek_sin') >>
               nvt.ops.AddMetadata(tags=[Tags.CONTINUOUS])
              )
    
weekday_cos= (sessiontime_weekday >> 
              (lambda col: get_cycled_feature_value_cos(col+1, 7)) >> 
              nvt.ops.Rename(name = 'et_dayofweek_cos') >>
              nvt.ops.AddMetadata(tags=[Tags.CONTINUOUS])
             )


# Compute Item recency: Define a custom Op 
class ItemRecency(nvt.ops.Operator):
    def transform(self, columns, gdf):
        for column in columns.names:
            col = gdf[column]
            item_first_timestamp = gdf['prod_first_event_time_ts']
            delta_days = (col - item_first_timestamp) / (60*60*24)
            gdf[column + "_age_days"] = delta_days * (delta_days >=0)
        return gdf

    def compute_selector(
        self,
        input_schema: Schema,
        selector: ColumnSelector,
        parents_selector: ColumnSelector,
        dependencies_selector: ColumnSelector,
    ) -> ColumnSelector:
        self._validate_matching_cols(input_schema, parents_selector, "computing input selector")
        return parents_selector

    def column_mapping(self, col_selector):
        column_mapping = {}
        for col_name in col_selector.names:
            column_mapping[col_name + "_age_days"] = [col_name]
        return column_mapping

    @property
    def dependencies(self):
        return ["prod_first_event_time_ts"]

    @property
    def output_dtype(self):
        return np.float64

recency_features = ['event_time_ts'] >> ItemRecency() 
recency_features_norm = (recency_features >> 
                         nvt.ops.LogOp() >> 
                         nvt.ops.Normalize(out_dtype=np.float32) >> 
                         nvt.ops.Rename(name='product_recency_days_log_norm')
                        )

time_features = (
    session_time +
    sessiontime_weekday +
    weekday_sin +
    weekday_cos +
    recency_features_norm
)


# Smoothing price long-tailed distribution and applying standardization
price_log = ['price'] >> nvt.ops.LogOp() >> nvt.ops.Normalize(out_dtype=np.float32) >> nvt.ops.Rename(name='price_log_norm')

# Relative price to the average price for the category_id
def relative_price_to_avg_categ(col, gdf):
    epsilon = 1e-5
    col = ((gdf['price'] - col) / (col + epsilon)) * (col > 0).astype(int)
    return col
    
avg_category_id_pr = ['category_id'] >> nvt.ops.JoinGroupby(cont_cols =['price'], stats=["mean"]) >> nvt.ops.Rename(name='avg_category_id_price')
relative_price_to_avg_category = (
    avg_category_id_pr >> 
    nvt.ops.LambdaOp(relative_price_to_avg_categ, dependency=['price']) >> 
    nvt.ops.Rename(name="relative_price_to_avg_categ_id") >>
    nvt.ops.AddMetadata(tags=[Tags.CONTINUOUS])
)


# Aggregate by session id and creates the sequential features

groupby_feats = ['event_time_ts', 'user_session'] + cat_feats + time_features + price_log + relative_price_to_avg_category

# Define Groupby Workflow
groupby_features = groupby_feats >> nvt.ops.Groupby(
    groupby_cols=["user_session"], 
    sort_cols=["event_time_ts"],
    aggs={
        'user_id': ['first'],
        'product_id': ["list", "count"],
        'category_code': ["list"],  
        'brand': ["list"], 
        'category_id': ["list"], 
        'event_time_ts': ["first"],
        'event_time_dt': ["first"],
        'et_dayofweek_sin': ["list"],
        'et_dayofweek_cos': ["list"],
        'price_log_norm': ["list"],
        'relative_price_to_avg_categ_id': ["list"],
        'product_recency_days_log_norm': ["list"]
        },
    name_sep="-")

groupby_features_list = groupby_features['product_id-list',
        'category_code-list',  
        'brand-list', 
        'category_id-list', 
        'et_dayofweek_sin-list',
        'et_dayofweek_cos-list',
        'price_log_norm-list',
        'relative_price_to_avg_categ_id-list',
        'product_recency_days_log_norm-list']

SESSIONS_MAX_LENGTH = 20 
MINIMUM_SESSION_LENGTH = 2

groupby_features_trim = groupby_features_list >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH, pad=True)

# calculate session day index based on 'timestamp-first' column
day_index = ((groupby_features['event_time_dt-first'])  >> 
             nvt.ops.LambdaOp(lambda col: (col - col.min()).dt.days +1) >> 
             nvt.ops.Rename(f = lambda col: "day_index") >>
             nvt.ops.AddMetadata(tags=[Tags.CATEGORICAL])
            )

sess_id = groupby_features['user_session'] >> nvt.ops.AddMetadata(tags=[Tags.CATEGORICAL])

selected_features = sess_id + groupby_features['product_id-count'] + groupby_features_trim + day_index

filtered_sessions = selected_features >> nvt.ops.Filter(f=lambda df: df["product_id-count"] >= MINIMUM_SESSION_LENGTH)

workflow = nvt.Workflow(filtered_sessions)

print(workflow.output_schema)
