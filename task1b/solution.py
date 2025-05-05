#!/usr/bin/env python
# coding: utf-8

# #### General guidance
# 
# This serves as a template which will guide you through the implementation of this task. It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# This is the jupyter notebook version of the template. For the python file version, please refer to the file `template_solution.py`.
# 
# First, we import necessary libraries:


import numpy as np
import pandas as pd

# Add any additional imports here (however, the task is solvable without using 
# any additional imports)
# import ...
import os

import sklearn as sk
import sklearn.model_selection
import sklearn.linear_model


#### Loading data
data = pd.read_csv("data/train.csv")
y = data["y"].to_numpy()
data = data.drop(columns=["Id", "y"])
# print a few data samples
print(data.head())
X = data.to_numpy()


"""
Transform the 5 input features of matrix X (x_i denoting the i-th component of X) 
into 21 new features phi(X) in the following manner:
5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
1 constant feature: phi_21(X)=1

Parameters
----------
X: matrix of floats, dim = (700,5), inputs with 5 features

Compute
----------
X_transformed: array of floats: dim = (700,21), transformed input with 21 features
"""

X_transformed = np.zeros((700, 21))

X_transformed[:, 0] = X[:, 0] # x1
X_transformed[:, 1] = X[:, 1] # x2
X_transformed[:, 2] = X[:, 2] # x3
X_transformed[:, 3] = X[:, 3] # x4
X_transformed[:, 4] = X[:, 4] # x5

X_transformed[:, 5] = X[:, 0]**2 # x1**2
X_transformed[:, 6] = X[:, 1]**2 # x2**2
X_transformed[:, 7] = X[:, 2]**2 # x3**2
X_transformed[:, 8] = X[:, 3]**2 # x4**2
X_transformed[:, 9] = X[:, 4]**2 # x5**2

X_transformed[:, 10] = np.exp(X[:, 0]) # exp(x1)
X_transformed[:, 11] = np.exp(X[:, 1]) # exp(x2)
X_transformed[:, 12] = np.exp(X[:, 2]) # exp(x3)
X_transformed[:, 13] = np.exp(X[:, 3]) # exp(x4)
X_transformed[:, 14] = np.exp(X[:, 4]) # exp(x5)

X_transformed[:, 15] = np.cos(X[:, 0]) # cos(x1)
X_transformed[:, 16] = np.cos(X[:, 1]) # cos(x2)
X_transformed[:, 17] = np.cos(X[:, 2]) # cos(x3)
X_transformed[:, 18] = np.cos(X[:, 3]) # cos(x4)
X_transformed[:, 19] = np.cos(X[:, 4]) # cos(x5)

X_transformed[:, 20] = 1 # bias

print(X[0:5, :])
print(X_transformed[0:5, :])

assert X_transformed.shape == (700, 21)


#### Fit data
"""
Use the transformed data points X_transformed and fit the linear regression on this 
transformed data. Finally, compute the weights of the fitted linear regression. 

Parameters
----------
X_transformed: array of floats: dim = (700,21), transformed input with 21 features
y: array of floats, dim = (700,), input labels)

Compute
----------
w: array of floats: dim = (21,), optimal parameters of linear regression
"""

train_size = 1

if train_size < 1:
    # Do test split for good measure
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_transformed, y, train_size=train_size, random_state=69)
else:
    X_train = X_transformed
    Y_train = y
    X_test = None


# Train model (no intercept as we already have that in our weights)
model = sk.linear_model.RidgeCV(fit_intercept=False, alphas=np.logspace(0, 10, 400))
model.fit(X_train, Y_train)

print(model.alpha_)

w = model.coef_
assert w.shape == (21,)


if X_test is not None:
    # Validate model
    test_preds = model.predict(X_test)
    diff = test_preds - Y_test
    RMSE = sklearn.metrics.mean_squared_error(Y_test, test_preds)**0.5

    print(f"RMSE: {RMSE}")


# Generate Output Files

# Save results in the required format
np.savetxt("./output.csv", w, fmt="%.12f")


