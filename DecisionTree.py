# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:34:51 2024

@author: Mahdi
"""

import torch
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

print(device)


col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv("data/diabetes.csv", header=0, names=col_names)

print(pima.head())

# We need to divide given columns into two types of variables 
#  dependent(or target variable) and independent variable(or feature variables).

#split dataset in features and target variable (Feature selection)
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable

# Split dataset into training set and test set (Splittig data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, random_state=1)

# Create Decision Tree classifer object
model = DecisionTreeClassifier()

# Train Decision Tree Classifer
model = model.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('images/diabetes.png')
Image(graph.create_png())


model = DecisionTreeClassifier(criterion="entropy", max_depth=3)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


dot_data = StringIO()
export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('images/diabetes2.png')
Image(graph.create_png())






