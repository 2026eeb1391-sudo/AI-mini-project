# 1. Import libraries
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


# 2. Load dataset
df = pd.read_csv("rainfall in india 1901-2015.csv")

print("First 5 rows:")
print(df.head())
print("\nDataset shape:")
print(df.shape)
print("\nDataset information:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# 3. Data Preprocessing 
data = df[['SUBDIVISION', 'YEAR', 'ANNUAL']].copy()
data = data.dropna(subset=['ANNUAL'])
data = data.dropna(subset=['SUBDIVISION', 'YEAR'])
data['YEAR'] = data['YEAR'].astype(int)
print("\nCleaned dataset:")
print(data.head())
print("\nRemaining missing values:")
print(data.isnull().sum())


# 4. EXPLORATORY DATA ANALYSIS

print("\nStatistical Summary:")
print(data.describe())
print("\nNumber of subdivisions:")
print(data['SUBDIVISION'].nunique())
print("\nCorrelation:")
print(data[['YEAR', 'ANNUAL']].corr())
plt.figure(figsize=(6, 4))
sns.heatmap(
    data[['YEAR', 'ANNUAL']].corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.show()


# 5. SELECT INPUT AND TARGET
X = data[['SUBDIVISION', 'YEAR']]
y = data['ANNUAL']


# 6. TRAIN-TEST SPLIT (CHRONOLOGICAL)
unique_years = sorted(data['YEAR'].unique())
split_index = int(len(unique_years) * 0.80)
split_year = unique_years[split_index - 1]
train_data = data[data['YEAR'] <= split_year]
test_data = data[data['YEAR'] > split_year]
X_train = train_data[['SUBDIVISION', 'YEAR']]
X_test = test_data[['SUBDIVISION', 'YEAR']]
y_train = train_data['ANNUAL']
y_test = test_data['ANNUAL']
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Last training year:", split_year)
print("First testing year:", test_data['YEAR'].min())


# 7. DATA PREPROCESSING FOR MODEL

preprocessor = ColumnTransformer(
    transformers=[
        ('subdivision',
         OneHotEncoder(handle_unknown='ignore'),
         ['SUBDIVISION'])
    ],
    remainder='passthrough'
)








