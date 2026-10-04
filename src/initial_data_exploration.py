import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

"""
The only thing you need from here is loading dataset.
Remaining methods calculate histograms, check correlations and basic statistics of the dataset.
"""

def load_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path, sep=',')
        print(f"Data loaded successfully from {file_path}\n")
        print(f"Part of loaded data: \n{data.head()}")
        print(f"Info about the data: \n{data.info()}")
        return data
    except FileNotFoundError as f:
        print(f"Error: The file at {file_path} was not found. Please check the path and try again.")
    except pd.errors.EmptyDataError as e:
        print(f"Error: The file at {file_path} is empty!")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")

def explore_data(dataframe: pd.DataFrame):
    print(f"Basic DataFrame statistics: \n{dataframe.describe()}\n")
    print(f"Some histogram visualizations of the data: \n{dataframe.hist(bins=30, figsize=(10, 10))}")
    plt.show()

def inital_correlation_analysis(dataframe: pd.DataFrame):
    """
    We assume that only non-numerical column is ocean_proximity, so we will drop it for correlation analysis.
    drop() doesn't affect the original dataframe.
    corr() meyhod calculates correlations between numerical columns
    """
    correlation_matrix = dataframe.drop(columns=['ocean_proximity']).corr()

    pairs = correlation_matrix.unstack()
    cleaned_pairs = pairs[pairs.index.get_level_values(0) > pairs.index.get_level_values(1)]
    corr_matrix_sorted = cleaned_pairs[cleaned_pairs.abs() > 0.5].sort_values(ascending=False)
    # We want to see only high correlations (e.g. > 0.6 excluding self correlations)
    print(f"Correlation matrix (correlations > 0.5): \n{corr_matrix_sorted}\n")

if __name__ == "__main__":
    file_path = Path("D:/California_housing_training/housing.csv")
    dataframe = load_data(file_path)
    #explore_data(dataframe)
    inital_correlation_analysis(dataframe)