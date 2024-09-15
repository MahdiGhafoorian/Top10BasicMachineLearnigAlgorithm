# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:56:39 2024

@author: Mahdi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


#################################### Common part ##############################

# Set random seed for reproducibility
np.random.seed(42)

# Generate random data with float values
num_samples = 200
hours_studied = np.random.uniform(1.0, 10.0, size=num_samples)  # Random hours between 1.0 and 10.0
hours_slept = np.random.uniform(1.0, 10.0, size=num_samples)    # Random hours between 4.0 and 8.0

# Determine if the student passed or failed
# Rule: Pass if both hours_studied and hours_slept are above their median values
median_studied = np.median(hours_studied)-1
median_slept = np.median(hours_slept)-3
passed = ((hours_studied > median_studied) & (hours_slept > median_slept)).astype(int)

# Create a DataFrame
data = pd.DataFrame({
    'Hours_Studied': hours_studied,
    'Hours_Slept': hours_slept,
    'Passed': passed
})

learning_rate = 0.01
num_iterations = 3000

###############################################################################
############# Binary Logistic regression manual implementation ################
###############################################################################

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def predict(features, weights):
  '''
  Returns 1D array of probabilities
  that the class label == 1
  '''
  z = np.dot(features, weights)
  return sigmoid(z)

def cost_function(features, labels, weights):
    '''
    Using Mean Absolute Error

    Features:(100,3)
    Labels (targets): (100,1) 
    Weights:(3,1)
    Returns 1D matrix of predictions
    Cost = (labels*log(predictions) + (1-labels)*log(1-predictions) ) / len(labels)
    '''
    observations = len(labels)

    predictions = predict(features, weights)

    #Take the error when label=1
    class1_cost = -labels*np.log(predictions)

    #Take the error when label=0
    class2_cost = (1-labels)*np.log(1-predictions)

    #Take the sum of both costs
    cost = class1_cost - class2_cost

    #Take the average cost
    cost = cost.sum() / observations

    return cost

def update_weights(features, labels, weights, lr):
    '''
    Vectorized Gradient Descent

    Features:(200, 3)
    Labels: (200, 1)
    Weights:(3, 1)
    '''
    N = len(features)

    #1 - Get Predictions
    predictions = predict(features, weights)

    #2 Transpose features from (200, 3) to (3, 200)
    # So we can multiply w the (200,1)  cost matrix.
    # Returns a (3,1) matrix holding 3 partial derivatives --
    # one for each feature -- representing the aggregate
    # slope of the cost function across all observations
    gradient = np.dot(features.T,  predictions - labels)

    #3 Take the average cost derivative for each feature
    gradient /= N

    #4 - Multiply the gradient by our learning rate
    gradient *= lr

    #5 - Subtract from our weights to minimize cost
    weights -= gradient

    return weights

def decision_boundary(prob):
  return 1 if prob >= .5 else 0

def classify(predictions):
  '''
  input  - N element array of predictions between 0 and 1
  output - N element array of 0s (False) and 1s (True)
  '''
  vectorized_decision_boundary = np.vectorize(decision_boundary)
  return vectorized_decision_boundary(predictions).flatten()

def train(features, labels, weights, lr, iters):
    cost_history = []

    for i in range(iters):
        weights = update_weights(features, labels, weights, lr)

        #Calculate error for auditing purposes
        cost = cost_function(features, labels, weights)
        cost_history.append(cost)

        # Log Progress
        if i % 1000 == 0:
            print("iter: "+str(i) + " cost: "+str(cost))

    return weights, cost_history

def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

weights = np.zeros((3, 1))
features = data[['Hours_Studied', 'Hours_Slept']].values
labels = data['Passed'].values.reshape(-1, 1)
bias = np.ones((len(data),1))
features = np.append(bias, features, axis=1)

weights, cost_history = train(features,labels,weights,learning_rate, num_iterations)

print(f'Weights = {weights}')

predictions = predict(features, weights)
acc = accuracy(predictions, labels)

print(f'Accuracy = {acc}')

# Plotting decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(data[data['Passed'] == 1]['Hours_Slept'],
            data[data['Passed'] == 1]['Hours_Studied'],              
            color='blue', marker='o', label='Passed', alpha=0.6)
plt.scatter(data[data['Passed'] == 0]['Hours_Slept'], 
            data[data['Passed'] == 0]['Hours_Studied'],             
            color='red', marker='s', label='Failed', alpha=0.6)

# Decision boundary
x_values = np.linspace(data['Hours_Slept'].min(), data['Hours_Slept'].max(), 1000)
y_values = -(weights[0] + weights[2] * x_values) / weights[1]
plt.plot(x_values, y_values, 'b--', label='Decision Boundary')

plt.xlabel('Hours Slept')
plt.ylabel('Hours Studied')
plt.title('Scatter Plot with Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(range(num_iterations),cost_history,color='blue')
plt.xlabel('iterations')
plt.ylabel('cost')
plt.title('cost over iteraton')
plt.grid(True)
plt.show()


###############################################################################
################ Binary Logistic regression Using package #####################
###############################################################################

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Prepare the data (X: features, y: labels)
X = data[['Hours_Studied', 'Hours_Slept']].values
y = data['Passed'].values.reshape(-1, 1)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
log_reg = LogisticRegression()

# Train the model on the training data
log_reg.fit(X_train, y_train.ravel())  # y_train.ravel() to flatten the array

# Predict on the test data
y_pred = log_reg.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plotting decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(data[data['Passed'] == 1]['Hours_Slept'],
            data[data['Passed'] == 1]['Hours_Studied'], 
            color='blue', marker='o', label='Passed', alpha=0.6)  # Circle marker
plt.scatter(data[data['Passed'] == 0]['Hours_Slept'], 
            data[data['Passed'] == 0]['Hours_Studied'],             
            color='red', marker='s', label='Failed', alpha=0.6)  # Square marker

# Decision boundary
x_values = np.linspace(data['Hours_Slept'].min(), data['Hours_Slept'].max(), 100)
y_values = -(log_reg.intercept_ + log_reg.coef_[0][1] * x_values) / log_reg.coef_[0][0]
plt.plot(x_values, y_values, 'b--', label='Decision Boundary')

plt.xlabel('Hours Slept')
plt.ylabel('Hours Studied')
plt.title('Scatter Plot with Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()


