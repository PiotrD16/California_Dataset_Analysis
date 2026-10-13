import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from split_data import rng, test_ratio, is_id_in_test_set, split_train_test_by_id
from initial_data_exploration import load_data
from pathlib import Path

import ClusterSimilarity

current_dir = Path(__file__).resolve().parent.parent
file_path = current_dir / 'dataset' / 'housing.csv'

# Load the dataset
dataset = load_data(file_path)
dataset.describe()

# helper functions (for calcuating coefficients - bedrooms coef, romms_per_family, people_per_house):
def column_ratio(matrix):
    return matrix[:, [0]] / matrix[:, [1]]
def ratio_name():
    return ['ratio']

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'), 
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
        )

log_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    FunctionTransformer(np.log, feature_names_out='one-to-one'),
    StandardScaler()
)

cluster_simil = ClusterSimilarity(n_cluster=10, gamma=0.1, random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())