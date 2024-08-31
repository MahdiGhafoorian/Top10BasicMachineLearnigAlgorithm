# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 12:50:24 2024

@author: Mahdi
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Example dataset (Assuming you have a CSV file or a pandas DataFrame)
# Let's say the dataset is in a CSV file named 'advertising.csv'
# with columns: 'TV', 'Radio', 'Newspaper', 'Sales'

# Load the dataset
df = pd.read_csv('data/advertising.csv')

# Define the feature columns and target column
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict the target values for the test set
y_pred = model.predict(X_test)


# Output the function learned for the learned weights (coefficients) and bias (intercept)
W_1, W_2, W_3 = model.coef_
bias = model.intercept_
print(f"Sales={W_1:.3f}TV+{W_2:.3f}Radio+.{W_3:.3f}Newspaper+{bias:.3f}")

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.3f}")
print(f"R-squared: {r2:.3f}")








