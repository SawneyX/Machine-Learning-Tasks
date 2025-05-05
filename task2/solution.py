# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    #TRAIN
    train_df = train_df.dropna(subset=['price_CHF'])   # Delete rows where price_CHF is missing: would introduce noise if the label was guessed with mean

    train_df = pd.get_dummies(train_df, columns=['season'], prefix='season')   #One hot encode season
    
    train_df.fillna(train_df.mean(), inplace=True)

    #TEST
    test_df = pd.get_dummies(test_df, columns=['season'], prefix='season')

    test_df.fillna(train_df.mean(), inplace=True)

        
        
    print("\n")
    print("Training data (after):")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print("\n")

    print("Test data (after):")
    print("Shape:", test_df.shape)
    print(test_df.head(2))
    print("\n")


    X_train = train_df.drop(['price_CHF'], axis=1)
    y_train = train_df['price_CHF']
    X_test = test_df


    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    """
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    }
    
    
    #SVR (SVM) initialize with rbf kernel trick
    svr = SVR(kernel='rbf')
    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best R2 Score:", grid_search.best_score_)

    # Initialize SVR with best found parameters
    best_svr = grid_search.best_estimator_
    
    
    # Fit the model to training data
    best_svr.fit(X_train, y_train) """
    
    
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)

    # Fit the model to the training data
    gpr.fit(X_train, y_train)
    

    # Predict
    """ y_pred_train = best_svr.predict(X_train)
    y_pred_test = best_svr.predict(X_test) """
    y_pred_train = gpr.predict(X_train)
    y_pred_test = gpr.predict(X_test)

    # Evaluate
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2 = r2_score(y_train, y_pred_train)

    print("Train RMSE:", train_rmse)
    print("Train r2 (higher better):", r2)
    y_pred = y_pred_test  

    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

