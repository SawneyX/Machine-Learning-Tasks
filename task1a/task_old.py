#!/usr/bin/env python
# coding: utf-8

# #### General guidance
# 
# This serves as a template which will guide you through the implementation of this task. It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# This is the jupyter notebook version of the template. For the python file version, please refer to the file `template_solution.py`.
# 
# First, we import necessary libraries:

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


# Add any additional imports here (however, the task is solvable without using 
# any additional imports)
# import ...
from sklearn.linear_model import Ridge
import os


# In[ ]:





#  #### Loading data

# In[3]:


# Pull data if not exists
DATA_PATH = 'data'
if not os.path.exists(DATA_PATH):
    !bash pull_data.sh
    
else:
    print("Data already fetched!") 


# In[3]:


df = pd.read_csv("data/train.csv")

Y = df.iloc[:, 0].to_numpy()
X = df.iloc[:, 1:].to_numpy()


# #### Calculating the average RMSE

# In[4]:


def calculate_RMSE(w, X, y):
    """This function takes test data points (X and y), and computes the empirical RMSE of 
    predicting y from X using a linear model with weights w. 

    Parameters
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression 
    X: matrix of floats, dim = (15,13), inputs with 13 features
    y: array of floats, dim = (15,), input labels

    Returns
    ----------
    RMSE: float: dim = 1, RMSE value
    """
    
    assert(w.shape == (13, ))
    assert(X.shape == (15, 13))
    assert(y.shape == (15, ))

    RMSE = 0

    
    y_actual = y
    y_predicted = np.dot(X, w)
    
    assert(y_predicted.shape == (15, ))

    
    RMSE = np.sqrt(np.square(np.subtract(y_actual, y_predicted)).mean()) #calcs RMSE 
    
    assert np.isscalar(RMSE)
    return RMSE


# #### Fitting the regressor

# In[5]:


def fit(X, y, lam):
    """
    This function receives training data points, then fits the ridge regression on this data
    with regularization hyperparameter lambda. The weights w of the fitted ridge regression
    are returned. 

    Parameters
    ----------
    X: matrix of floats, dim = (135,13), inputs with 13 features
    y: array of floats, dim = (135,), input labels)
    lam: float. lambda parameter, used in regularization term

    Returns
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression
    """
    assert(X.shape[1] == 13)

    model = Ridge(alpha=lam, fit_intercept=False)
    model.fit(X, y)
    
    w = model.coef_
    
    assert w.shape == (13,)
    return w


# #### Performing computation

# In[6]:


"""
Main cross-validation loop, implementing 10-fold CV. In every iteration 
(for every train-test split), the RMSE for every lambda is calculated, 
and then averaged over iterations.

Parameters
---------- 
X: matrix of floats, dim = (150, 13), inputs with 13 features
y: array of floats, dim = (150, ), input labels
lambdas: list of floats, len = 5, values of lambda for which ridge regression is fitted and RMSE estimated
n_folds: int, number of folds (pieces in which we split the dataset), parameter K in KFold CV

Compute
----------
avg_RMSE: array of floats: dim = (5,), average RMSE value for every lambda
"""

# The function calculating the average RMSE
lambdas = [0.1, 1, 10, 100, 200]
n_folds = 10

RMSE_mat = np.zeros((n_folds, len(lambdas)))

k_fold = KFold(n_splits=n_folds)

for i, (train_index, test_index) in enumerate(k_fold.split(X)):

    fold_X_train, fold_X_test = X[train_index], X[test_index]
    fold_y_train, fold_y_test = Y[train_index], Y[test_index]

    for j in range(len(lambdas)):
    
        fold_w = fit(fold_X_train, fold_y_train, lambdas[j])
        fold_rmse = calculate_RMSE(fold_w, fold_X_test, fold_y_test)
        RMSE_mat[i][j] = fold_rmse
        

avg_RMSE = np.mean(RMSE_mat, axis=0) # avg_RMSE: array of floats: dim = (5,), average RMSE value for every lambda
display(avg_RMSE)
assert avg_RMSE.shape == (5,)
print(RMSE_mat)
print(avg_RMSE)


# # Create Outputs
# 

# In[7]:


# Save results in the required format
np.savetxt("./output.csv", avg_RMSE, fmt="%.12f")


# In[12]:


get_ipython().system('jupyter nbconvert --to script task.ipynb')


# In[ ]:




