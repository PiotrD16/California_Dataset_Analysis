import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def create_scatter_matrix(dataframe: pd.DataFrame):
    # These are the most correlated attributes with median_house_value (target variable)
    attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']

    pd.plotting.scatter_matrix(dataframe[attributes], figsize=(12, 8))
    plt.suptitle("Scatter Matrix of Selected Attributes", fontsize=16)
    plt.show()

def longitude_latitude_scatter_plot(dataframe: pd.DataFrame):
    plt.figure(figsize=(10, 7))
    plt.scatter(dataframe['longitude'], 
                dataframe['latitude'], 
                alpha=0.3, 
                c=dataframe['median_house_value'], 
                cmap='viridis', 
                s=dataframe['population']/100, 
                linewidth=0.5
                )
    plt.colorbar(label='Median House Value')
    plt.title("Longitude vs Latitude Scatter Plot")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

if __name__ == "__main__":
    file_path = Path("D:/California_housing_training/housing.csv")
    dataframe = pd.read_csv(file_path, sep=',')
    create_scatter_matrix(dataframe)
    longitude_latitude_scatter_plot(dataframe)
    input("Press Enter to exit...")