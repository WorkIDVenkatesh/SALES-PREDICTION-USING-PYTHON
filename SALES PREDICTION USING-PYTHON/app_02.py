# SALES PREDICTION USING PYTHON
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("1. Loading the advertising dataset...")
try:
    df = pd.read_csv('advertising.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'advertising.csv' not found. Please make sure the file is in the same directory.")
    exit()
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\n2. Defining features and target variable...")
X = df.drop('Sales', axis=1)
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into training set (shape: {X_train.shape}) and testing set (shape: {X_test.shape}).")
print("\n3. Training the Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete.")
print("\n4. Making predictions and evaluating the model...")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.6}, line_kws={'color':'red', 'linewidth':2})
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs. Predicted Sales")
plt.grid(True)
plt.show()
print("\nScript execution finished. Check the plot for a visual representation of the model's performance.")