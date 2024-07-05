# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 17:54:25 2024

@author: rodri
"""

#%% Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#%% Load the data
X_train = pd.read_csv("C:/Users/rodri/Documents/ds/ml/dados/regressao/X_training.csv")
y_train = pd.read_csv("C:/Users/rodri/Documents/ds/ml/dados/regressao/y_training.csv").values.ravel()  # Ensure y is 1D array
X_val = pd.read_csv("C:/Users/rodri/Documents/ds/ml/dados/regressao/X_validation.csv")
y_val = pd.read_csv("C:/Users/rodri/Documents/ds/ml/dados/regressao/y_val.csv").values.ravel()
X_test = pd.read_csv("C:/Users/rodri/Documents/ds/ml/dados/regressao/X_test.csv")
y_test = pd.read_csv("C:/Users/rodri/Documents/ds/ml/dados/regressao/y_test.csv").values.ravel()

#%% Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#%% Function to calculate performance metrics
def calculate_metrics(model, X, y):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    return r2, mse, rmse, mae, mape

#%% List of models with their parameters
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(max_depth=10),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=10),
    "Polynomial Regression": Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())]),
    "Lasso Regression": Lasso(alpha=1.0),
    "Ridge Regression": Ridge(alpha=1.0),
    "ElasticNet Regression": ElasticNet(alpha=1.0, l1_ratio=0.5),
    "Polynomial Lasso Regression": Pipeline([('poly', PolynomialFeatures(degree=2)), ('lasso', Lasso(alpha=1.0))]),
    "Polynomial Ridge Regression": Pipeline([('poly', PolynomialFeatures(degree=2)), ('ridge', Ridge(alpha=1.0))]),
    "Polynomial ElasticNet Regression": Pipeline([('poly', PolynomialFeatures(degree=2)), ('elasticnet', ElasticNet(alpha=1.0, l1_ratio=0.5))]),
}

#%% Initialize lists to store results
results_train = []
results_val = []
results_test = []

#%% Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    
    # Evaluate on training data
    metrics_train = calculate_metrics(model, X_train, y_train)
    results_train.append((name, *metrics_train))
    
    # Evaluate on validation data
    metrics_val = calculate_metrics(model, X_val, y_val)
    results_val.append((name, *metrics_val))
    
    # Evaluate on test data
    metrics_test = calculate_metrics(model, X_test, y_test)
    results_test.append((name, *metrics_test))

#%% Create DataFrames for results
columns = ["Algorithm", "R2", "MSE", "RMSE", "MAE", "MAPE"]
df_train = pd.DataFrame(results_train, columns=columns)
df_val = pd.DataFrame(results_val, columns=columns)
df_test = pd.DataFrame(results_test, columns=columns)

#%% Display the results
print("Performance on Training Data")
print(df_train)
print("\nPerformance on Validation Data")
print(df_val)
print("\nPerformance on Test Data")
print(df_test)

#%%
df_train.to_csv(r'C:\Users\rodri\Documents\ds\ml\code\train_performance_reg.csv', index=False)
df_val.to_csv(r'C:\Users\rodri\Documents\ds\ml\code\validation_performance_reg.csv', index=False)
df_test.to_csv(r'C:\Users\rodri\Documents\ds\ml\code\test_performance_reg.csv', index=False)