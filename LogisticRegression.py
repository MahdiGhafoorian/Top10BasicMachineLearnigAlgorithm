# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:56:39 2024

@author: Mahdi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split



##### Select module to Run #####
# 1 for Simple regression manual implementation
# 2 for Simple regression using package
# 3 for Multivariable regression manual implementation
# 4 for Multivariable regression using package
Module = 3

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

# Generate random multiclass labels (e.g., 3 classes: 0, 1, 2)
# Students with low study/sleep hours are 0, mid-range 1, and high-range 2.
labels = np.zeros(num_samples, dtype=int)
labels[hours_studied + hours_slept > 14] = 2  # Class 2: high study and sleep
labels[(hours_studied + hours_slept > 10) & (hours_studied + hours_slept <= 14)] = 1  # Class 1: mid-range
labels[hours_studied + hours_slept <= 10] = 0  # Class 0: low study and sleep

# Create DataFrame
multiclass_data = pd.DataFrame({
    'Hours_Studied': hours_studied,
    'Hours_Slept': hours_slept,
    'Class': labels
})

learning_rate = 0.01
num_iterations = 3000

###############################################################################
############# Binary Logistic regression manual implementation ################
###############################################################################

if Module == 1:
    
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

elif Module == 2:
    
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
    
###############################################################################
############ Multiclass Logistic regression Manual implementation #############
###############################################################################

elif Module == 3:
    # Prepare the data
    X = multiclass_data[['Hours_Studied', 'Hours_Slept']].values
    y = multiclass_data['Class'].values
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Step 3: Implement Multiclass Logistic Regression (Softmax)
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def compute_cost(X, y, theta):
        m = X.shape[0]
        predictions = softmax(X @ theta)
        log_likelihood = -np.log(predictions[np.arange(m), y])
        return np.sum(log_likelihood) / m
    
    def gradient_descent(X, y, theta, alpha, num_iters):
        m = X.shape[0]
        costs = []
        for _ in range(num_iters):
            predictions = softmax(X @ theta)
            gradient = (1 / m) * (X.T @ (predictions - np.eye(theta.shape[1])[y]))
            theta -= alpha * gradient
            costs.append(compute_cost(X, y, theta))
        return theta, costs
    
    # Initialize parameters
    num_classes = len(np.unique(y))
    X_train_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))  # Add bias term
    theta = np.zeros((X_train_bias.shape[1], num_classes))  # Initialize weights
    
    # Set learning parameters
    alpha = 0.1
    num_iters = 3000
    
    # Train the model
    theta, costs = gradient_descent(X_train_bias, y_train, theta, alpha, num_iters)
    
    # Step 4: Predict and Calculate Accuracy
    def predict(X, theta):
        X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.argmax(softmax(X_bias @ theta), axis=1)
    
    y_pred = predict(X_test, theta)
    accuracy = np.mean(y_pred == y_test) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Step 5: Plot Decision Boundaries and Data Points
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of each class in the test data
    plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='red', marker='s', label='Class 0')
    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='blue', marker='o', label='Class 1')
    plt.scatter(X_test[y_test == 2][:, 0], X_test[y_test == 2][:, 1], color='green', marker='^', label='Class 2')
    
    # Plot decision boundaries
    x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_values = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    xx, yy = np.meshgrid(x_values, y_values)
    # concat flatten data
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_bias = np.hstack((np.ones((grid.shape[0], 1)), grid))
    probs = softmax(np.dot(grid_bias, theta))
    Z = np.argmax(probs, axis=1).reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.xlabel('Hours Studied')
    plt.ylabel('Hours Slept')
    plt.title('Multiclass Logistic Regression Decision Boundaries')
    plt.legend()
    plt.grid(True)
    plt.show()

###############################################################################
############# Multiclass Logistic regression Using package ####################
###############################################################################

elif Module == 4:
    
    # Prepare the data (X: features, y: multiclass labels)
    X = multiclass_data[['Hours_Studied', 'Hours_Slept']].values
    y = multiclass_data['Class'].values
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create a logistic regression model for multiclass classification (softmax)
    log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    # multinomial means softmax
    
    # Train the model on the data
    log_reg.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = log_reg.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Display the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (Test Set):")
    print(conf_matrix)
    
    # Scatter plot of each class in the training data
    plt.scatter(multiclass_data[multiclass_data['Class'] == 0]['Hours_Slept'], 
                multiclass_data[multiclass_data['Class'] == 0]['Hours_Studied'],             
                color='red', marker='s', label='Class 0')
    plt.scatter(multiclass_data[multiclass_data['Class'] == 1]['Hours_Slept'], 
                multiclass_data[multiclass_data['Class'] == 1]['Hours_Studied'], 
                color='blue', marker='o', label='Class 1')
    plt.scatter(multiclass_data[multiclass_data['Class'] == 2]['Hours_Slept'], 
                multiclass_data[multiclass_data['Class'] == 2]['Hours_Studied'], 
                color='green', marker='^', label='Class 2')
    
    # Plot decision boundaries
    x_values = np.linspace(data['Hours_Studied'].min(), data['Hours_Studied'].max(), 100)
    y_values = np.linspace(data['Hours_Slept'].min(), data['Hours_Slept'].max(), 100)
    xx, yy = np.meshgrid(x_values, y_values)
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict class for each point in the grid
    Z = log_reg.predict(grid)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(-0.5, 3.5, 1), cmap=plt.cm.coolwarm)
    
    plt.xlabel('Hours Slept')
    plt.ylabel('Hours Studied')
    plt.title('Multiclass Logistic Regression Decision Boundaries')
    plt.legend()
    plt.grid(True)
    plt.show()







