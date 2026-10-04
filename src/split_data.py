import numpy as np
import pandas as pd
from zlib import crc32
from initial_data_exploration import load_data
from pathlib import Path

rng = np.random.default_rng(seed=42) # seed is set for reproducibility of the results
test_ratio = 0.2 # 80% - training data, 20% - test data

def is_id_in_test_set(id, test_ratio):
    return crc32(np.int64(id)) < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, column):
    ids = data[column] # column indicates what should be treated as an id (e.g. index)
    in_test_set = ids.apply(lambda id: is_id_in_test_set(id, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set] # train_set, test_set

if __name__ == "__main__":
    filepath = Path("D:/California_housing_training/housing.csv")
    data = load_data(filepath)
    dataset_with_id = data.reset_index() # we will use index as an id for splitting the dataset
    training_set, test_set = split_train_test_by_id(dataset_with_id, test_ratio, "index")
    print(f"Training set size: {len(training_set)}\nTest set size: {len(test_set)}")