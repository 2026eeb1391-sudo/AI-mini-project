# 1. Import libraries
import pandas as pd
data = pd.read_csv("rainfall in india 1901-2015.csv")
print(data.head)
pip install pandas numpy matplotlib seaborn scikit-learn

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

# 8. CREATE LINEAR REGRESSION MODEL

model = Pipeline([
    ('preprocessing', preprocessor),
    ('regression', LinearRegression())
])


# 9. TRAINING FUNCTION

def train_model():
    model.fit(X_train, y_train)
    print("\nModel training completed successfully!")


# 10. TESTING FUNCTION

def test_model():
    predictions = model.predict(X_test)
    return predictions


# 11. TRAIN THE MODEL

train_model()

# 12. MAKE PREDICTIONS ON TEST DATA

y_pred = test_model()

# 13. EVALUATION METRICS

def evaluate_model(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    mse = mean_squared_error(actual, predicted)

    rmse = np.sqrt(mse)

    r2 = r2_score(actual, predicted)

    print("\n MODEL EVALUATION ")

    print("Mean Absolute Error (MAE):", round(mae, 2))

    print("Mean Squared Error (MSE):", round(mse, 2))

    print("Root Mean Squared Error (RMSE):", round(rmse, 2))

    print("R2 Score:", round(r2, 4))

    print("Accuracy-like R2 percentage:", round(r2 * 100, 2), "%")


evaluate_model(y_test, y_pred)

# 14. TRAINING PERFORMANCE

train_predictions = model.predict(X_train)

print("\n TRAINING PERFORMANCE ")

evaluate_model(y_train, train_predictions)

# 15. ACTUAL VS PREDICTED VALUES

results = pd.DataFrame({
    'Subdivision': X_test['SUBDIVISION'].values,
    'Year': X_test['YEAR'].values,
    'Actual Rainfall': y_test.values,
    'Predicted Rainfall': y_pred
})

print("\nActual vs Predicted Rainfall:")
print(results.head(15))


# 16. GRAPH 1: TRAINING DATA

plt.figure(figsize=(10, 5))

plt.scatter(
    y_train,
    train_predictions,
    alpha=0.5
)

plt.xlabel("Actual Rainfall (mm)")
plt.ylabel("Predicted Rainfall (mm)")
plt.title("Training: Actual vs Predicted Rainfall")

plt.show()

# 17. GRAPH 2: TESTING DATA

plt.figure(figsize=(10, 5))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.xlabel("Actual Rainfall (mm)")
plt.ylabel("Predicted Rainfall (mm)")
plt.title("Testing: Actual vs Predicted Rainfall")

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    'r--'
)

plt.show()



# 18. GRAPH 3: RESIDUAL PLOT

residuals = y_test - y_pred

plt.figure(figsize=(10, 5))

plt.scatter(y_pred, residuals, alpha=0.5)

plt.axhline(y=0, color='red', linestyle='--')

plt.xlabel("Predicted Rainfall (mm)")
plt.ylabel("Residuals (mm)")
plt.title("Residual Plot")

plt.show()



# 19. GRAPH 4: ACTUAL VS PREDICTED FOR SAMPLE TEST RECORDS

comparison = results.head(30)

plt.figure(figsize=(12, 5))

plt.plot(
    range(len(comparison)),
    comparison['Actual Rainfall'],
    marker='o',
    label='Actual Rainfall'
)

plt.plot(
    range(len(comparison)),
    comparison['Predicted Rainfall'],
    marker='o',
    label='Predicted Rainfall'
)

plt.xlabel("Test Record")
plt.ylabel("Annual Rainfall (mm)")
plt.title("Actual vs Predicted Rainfall Comparison")

plt.legend()
plt.show()


print("\n========== RAINFALL PREDICTION ==========")

subdivisions = sorted(data['SUBDIVISION'].unique())

print("\nAvailable subdivisions:")
for s in subdivisions:
    print(s)

while True:

    subdivision = input(
        "\nEnter subdivision name: "
    ).strip().upper()

    if subdivision in subdivisions:
        break

    print("Invalid subdivision! Please enter an available name.")

year = int(input("Enter year for prediction: "))

# Create input DataFrame
new_data = pd.DataFrame({
    'SUBDIVISION': [subdivision],
    'YEAR': [year]
})

# Predict rainfall
predicted_rainfall = model.predict(new_data)


print("RAINFALL PREDICTION RESULT")
print("Subdivision:", subdivision)
print("Year:", year)
print("Predicted Annual Rainfall:",
      round(predicted_rainfall[0], 2), "mm")













