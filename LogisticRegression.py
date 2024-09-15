# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:56:39 2024

@author: Mahdi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

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

weights = np.zeros((3, 1))
features = data[['Hours_Studied', 'Hours_Slept']].values
labels = data['Passed'].values.reshape(-1, 1)
bias = np.ones((len(data),1))
features = np.append(bias, features, axis=1)

weights, cost_history = train(features,labels,weights,learning_rate, num_iterations)

print(weights)

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

sys.exit(0)

###############################################################################
 # Prepare the data
X = data[['Hours_Studied', 'Hours_Slept']].values
y = data['Passed'].values

# Add a bias column with ones
X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add column of ones for bias term

# Initialize parameters
weights = np.zeros(X.shape[1])  # One weight for each feature plus bias


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute cost function
def compute_cost(y, y_pred):
    m = len(y)
    return -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / m

cost_history = []

# Training loop
for _ in range(num_iterations):
    # Forward propagation
    z = np.dot(X, weights)
    y_pred = sigmoid(z)
    
    # Compute cost
    cost = compute_cost(y, y_pred)
    cost_history.append(cost)
    
    # Backward propagation
    gradient = np.dot(X.T, (y_pred - y)) / len(y)
    weights -= learning_rate * gradient
    
    # Optionally print cost every 100 iterations
    if _ % 100 == 0:
        print(f"Iteration {_}: Cost {cost}")

# Display final weights
print("Final weights:", weights)

# Predict using the model
# def predict(X, weights):
#     z = np.dot(X, weights)
#     return sigmoid(z) >= 0.5



# Plotting decision boundary
# plt.figure(figsize=(10, 6))
# plt.scatter(data[data['Passed'] == 1]['Hours_Studied'], 
#             data[data['Passed'] == 1]['Hours_Slept'], 
#             color='blue', marker='o', label='Passed', alpha=0.6)  # Circle marker
# plt.scatter(data[data['Passed'] == 0]['Hours_Studied'], 
#             data[data['Passed'] == 0]['Hours_Slept'], 
#             color='red', marker='s', label='Failed', alpha=0.6)  # Square marker

# # Decision boundary
# x_values = np.linspace(data['Hours_Studied'].min(), data['Hours_Studied'].max(), 100)
# y_values = -(weights[0] + weights[1] * x_values) / weights[2]
# plt.plot(x_values, y_values, 'b--', label='Decision Boundary')

# plt.xlabel('Hours Studied')
# plt.ylabel('Hours Slept')
# plt.title('Scatter Plot with Decision Boundary')
# plt.legend()
# plt.grid(True)
# plt.show()

plt.figure(figsize=(10,6))
plt.plot(range(num_iterations),cost_history,color='blue')
plt.xlabel('iterations')
plt.ylabel('cost')
plt.title('cost over iteraton')
plt.grid(True)
plt.show()


###############################################################################







sys.exit(1)
