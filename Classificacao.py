# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#%% dados

X_train = pd.read_csv("C:/Users/rodri/Documents/ds/ml/dados/classificacao/X_training.csv")
y_train = pd.read_csv("C:/Users/rodri/Documents/ds/ml/dados/classificacao/y_training.csv")
y_validation = pd.read_csv("C:/Users/rodri/Documents/ds/ml/dados/classificacao/y_validation.csv")
X_validation = pd.read_csv ("C:/Users/rodri/Documents/ds/ml/dados/classificacao/X_validation.csv")
y_test = pd.read_csv("C:/Users/rodri/Documents/ds/ml/dados/classificacao/y_test.csv")
X_test = pd.read_csv("C:/Users/rodri/Documents/ds/ml/dados/classificacao/X_test.csv")

#%% Define a function to evaluate the model

def evaluate_model(model, X_train, y_train, X_validation, y_validation, X_test, y_test):
    model.fit(X_train, y_train)
    
    datasets = {
        'Training': (X_train, y_train),
        'Validation': (X_validation, y_validation),
        'Test': (X_test, y_test)
    }
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    results = {dataset: {metric: 0 for metric in metrics} for dataset in datasets}
    
    for dataset_name, (X, y) in datasets.items():
        y_pred = model.predict(X)
        results[dataset_name]['Accuracy'] = accuracy_score(y, y_pred)
        results[dataset_name]['Precision'] = precision_score(y, y_pred, average='weighted')
        results[dataset_name]['Recall'] = recall_score(y, y_pred, average='weighted')
        results[dataset_name]['F1-Score'] = f1_score(y, y_pred, average='weighted')
    
    return results

#%% Define models
models = {
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression()
}

#%%
all_results = {model_name: evaluate_model(model, X_train, y_train, X_validation, y_validation, X_test, y_test) for model_name, model in models.items()}

#%% Convert results to a pandas DataFrame for better visualization
results_df = pd.DataFrame(all_results).T.stack().unstack(0)
print(results_df)

#%%
results_df.to_csv(r'C:\Users\rodri\Documents\ds\ml\code\results_df_class.csv', index=False)