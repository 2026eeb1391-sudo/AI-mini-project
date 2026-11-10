import pandas as pd
data = pd.read_csv("rainfall in india 1901-2015.csv")
print(data.head)
!pip install pandas numpy matplotlib seaborn scikit-learn



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
df = pd.read_csv("rainfall in india 1901-2015.csv")

print("First 5 rows:")
print(df.head())
print("\nDataset shape:")
print(df.shape)
print("\nDataset information:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())


data = df[['SUBDIVISION', 'YEAR', 'ANNUAL']].copy()
data = data.dropna(subset=['ANNUAL'])
data = data.dropna(subset=['SUBDIVISION', 'YEAR'])
data['YEAR'] = data['YEAR'].astype(int)
print("\nCleaned dataset:")
print(data.head())
print("\nRemaining missing values:")
print(data.isnull().sum())







