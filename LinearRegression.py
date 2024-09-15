# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 12:50:24 2024

@author: Mahdi
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import sys

##### Select module to Run #####
# 1 for Simple regression manual implementation
# 2 for Simple regression using package
# 3 for Multivariable regression manual implementation
# 4 for Multivariable regression using package
Module = 3

# Example dataset (Assuming you have a CSV file or a pandas DataFrame)
# Let's say the dataset is in a CSV file named 'advertising.csv'
# with columns: 'TV', 'Radio', 'Newspaper', 'Sales'

# Load the dataset
df = pd.read_csv('data/advertising.csv')

###############################################################################
################ Simple regression manual implementation ######################
###############################################################################

if Module == 1:
    
    def cost_function(feature, target, weight, bias):
        companies = len(feature)
        total_error = 0.0
        for i in range(companies):
            total_error += (target[i] - (weight*feature[i] + bias))**2
        return total_error / companies
    
    def update_weights(feature, target, weight, bias, learning_rate):
        weight_deriv = 0
        bias_deriv = 0
        companies = len(feature)
    
        for i in range(companies):
            # Calculate partial derivatives
            # -2x(y - (mx + b))
            weight_deriv += -2*feature[i] * (target[i] - (weight*feature[i] + bias))
    
            # -2(y - (mx + b))
            bias_deriv += -2*(target[i] - (weight*feature[i] + bias))
    
        # We subtract because the derivatives point in direction of steepest ascent
        weight -= (weight_deriv / companies) * learning_rate
        bias -= (bias_deriv / companies) * learning_rate
    
        return weight, bias
    
    def train(feature, target, weight, bias, learning_rate, iters):
        cost_history = []
    
        for i in range(iters):
            weight,bias = update_weights(feature, target, weight, bias, learning_rate)
    
            #Calculate cost for auditing purposes
            cost = cost_function(feature, target, weight, bias)
            cost_history.append(cost)
    
            # Log Progress
            if i % 10 == 0:
                print("iter={:d}    weight={:.2f}    bias={:.4f}    cost={:.2}".format(i, weight, bias, cost))
    
        return weight, bias, cost_history
        
    # Define the feature columns and target column
    X = df['Radio']
    y = df['Sales']
    
    
    weight, bias, cost_history = train(X, y, weight = 0.03, bias = 0, 
                                       learning_rate=0.001, iters=100)
    
    print(f"Sales = {weight:.3f} Radio + {bias:.3f}")


###############################################################################
####################### Simple regression using package #######################
###############################################################################

elif Module == 2:
        
    # Define the feature columns and target column
    X = df[['Radio']]
    y = df['Sales']
    
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Creating and training the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predicting the test set results
    y_pred = model.predict(X_test)
    
    # Output the function learned for the learned weights (coefficients) and bias (intercept)
    weight = model.coef_
    bias = model.intercept_
    print(f"Sales = {weight[0]:.3f} Radio + {bias:.3f}")
    
    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("MSE is:", mse)
    print("R2 score is:", r2)


###############################################################################
############# Multivariable regression manual implementation ##################
###############################################################################

elif Module == 3:
    
    # Normalizing data
    # For each feature column {
    #     #1 Subtract the mean of the column (mean normalization)
    #     #2 Divide by the range of the column (feature scaling)
    # }
    
    
    def normalize(features):
        """
        features     -   (200, 3)
        features.T   -   (3, 200)
    
        We transpose the input matrix, swapping
        cols and rows to make vector math easier
        """
    
        for feature in features.T:
            fmean = np.mean(feature)
            frange = np.amax(feature) - np.amin(feature)
    
            #Vector Subtraction
            feature -= fmean
    
            #Vector Division
            feature /= frange
    
        return features
    
    
    def predict(features, weights):
        """
        features - (200, 3)
        weights - (3, 1)
        predictions - (200,1)
        """
        #predictions = np.dot(features[:,:-1], weights)
        predictions = np.dot(features, weights)
        return predictions
    
    def cost_function(features, targets, weights):
        """
        features:(200,3)
        targets: (200,1)
        weights:(3,1)
        returns average squared error among predictions
        """
        N = len(targets)
    
        predictions = predict(features, weights)
    
        # Matrix math lets use do this without looping
        sq_error = (predictions.flatten() - targets)**2
    
        # Return average squared error among predictions
        return 1.0/(2*N) * sq_error.sum()
    
    def update_weights(features, targets, weights, lr):
        '''
        Features:(200, 3)
        Targets: (200, 1)
        Weights:(3, 1)
        '''
        predictions = predict(features, weights)
    
        #Extract our features
        x1 = features[:,0]
        x2 = features[:,1]
        x3 = features[:,2]
    
        # Use dot product to calculate the derivative for each weight
        d_w1 = -x1.dot(targets - predictions)
        d_w2 = -x2.dot(targets - predictions)
        d_w3 = -x3.dot(targets - predictions)
    
        # Multiply the mean derivative by the learning rate
        # and subtract from our weights (remember gradient points in direction of steepest ASCENT)
        weights[0][0] -= (lr * np.mean(d_w1))
        weights[1][0] -= (lr * np.mean(d_w2))
        weights[2][0] -= (lr * np.mean(d_w3))
    
        return weights
    
    def update_weights_vectorized(X, targets, weights, lr):
        '''
        gradient = X.T * (predictions - targets) / N
        X: (200, 3)
        Targets: (200, 1)
        Weights: (3, 1)
        '''
        companies = len(X)
    
        #1 - Get Predictions
        predictions = predict(X, weights)
    
        #2 - Calculate error/loss
        error = targets - predictions.flatten()
    
        #3 Transpose features from (200, 3) to (3, 200)
        # So we can multiply w the (200,1)  error matrix.
        # Returns a (3,1) matrix holding 3 partial derivatives --
        # one for each feature -- representing the aggregate
        # slope of the cost function across all observations
        gradient = np.dot(-X.T,  error)
        
    
        # 4 Take the average error derivative for each feature
        gradient /= companies
    
        # 5 - Multiply the gradient by our learning rate
        gradient *= lr
    
        # 6 - Subtract from our weights to minimize cost
        weights -= gradient.reshape(-1, 1)        
    
        return weights
    
    def train(features, targets, bias, weights, learning_rate, iters):
        cost_history = []
    
        for i in range(iters):
            weights = update_weights_vectorized(features, targets, weights, learning_rate)
    
            #Calculate cost for auditing purposes
            cost = cost_function(features, targets, weights)
            cost_history.append(cost)
    
            # Log Progress
            if i % 10 == 0:
                weights_formatted = " ".join("{:.2f}".format(w[0]) for w in weights[1:])
                bias = " ".join("{:.2f}".format(w[0]) for w in weights[:1])
                print("iter={:d}   weights={}   bias={:.2}  cost={:.2}".format(i, weights_formatted, bias, cost))
    
        return weights, cost_history
    
    # initialize weights 
    W_bias = 0.0
    W1 = 0.0
    W2 = 0.0
    W3 = 0.0
    weights = np.array([
        [W_bias],
        [W1],
        [W2],
        [W3]
    ])
    
    
        
    # Define the feature columns and target column
    features = df[['TV', 'Radio', 'Newspaper']].values
    targets = df['Sales'].values

    features = normalize(features)
    
    bias = np.ones(shape=(len(features),1))
    features = np.append(bias, features, axis=1)
    
    
    weights, cost_history = train(features, targets, bias, weights,
                                       learning_rate=0.001, iters=100)
    bias = weights[0][0]
    print(f"Sales = {weights[1][0]:.3f} TV + {weights[2][0]:.3f} Radio + {weights[3][0]:.3f} Newspaper + {bias:.3f}")
        
    # Predicting new values
    def Predicting(X_new, weights):
        X_new = (X_new - np.mean(features[:, 1:], axis=0)) / np.std(features[:, 1:], axis=0)  # Scale new data (excluding the bias term)
        X_new = np.insert(X_new,0,1) # Add bias term
        return X_new.dot(weights)
    
    # Example prediction
    new_data = np.array([230, 37, 69])  # New TV, Radio, Newspaper data
    predicted_sales = Predicting(new_data, weights)
    print(f"Predicted Sales for new data: {predicted_sales}")


###############################################################################
################# Multivariable regression using package ######################
###############################################################################
elif Module == 4:
        
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














