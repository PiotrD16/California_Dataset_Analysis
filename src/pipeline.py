import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from split_data import test_ratio, split_train_test_by_id
from initial_data_exploration import load_data
from pathlib import Path

from ClusterSimilarity import ClusterSimilarity

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

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=0.1, random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())

# Pipeline dla kategorii:
cat_attr = ['ocean_proximity']
cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))

# Gotowy pipeline do odpalenia
preprocessing = ColumnTransformer([
    ('bedrooms_coeff', ratio_pipeline(), ['total_bedrooms', 'total_rooms']),
    ('rooms_per_family', ratio_pipeline(), ['total_rooms', 'households']),
    ('people_per_house', ratio_pipeline(), ['population', 'households']),
    ('log', log_pipeline, ['total_bedrooms', 'total_rooms', 'population', 'households', 'median_income']),
    ('geographical', cluster_simil, ['latitude', 'longitude']),
    ('cat', cat_pipeline, make_column_selector(dtype_include=object))
], remainder=default_num_pipeline)

# Pobór danych:
dataset = dataset.reset_index()
train_set, test_set = split_train_test_by_id(dataset, test_ratio, 'index')

y_train = train_set['median_house_value'].copy()
y_test = test_set['median_house_value'].copy()

X_train = train_set.drop(columns=['index', 'median_house_value'])
X_test = test_set.drop(columns=['index', 'median_house_value'])

# Odpalam Pipeline:
housing_prepared = preprocessing.fit_transform(X_train)
housing_test_prepared = preprocessing.fit_transform(X_test)