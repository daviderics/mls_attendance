"""
The functions in this file are used in modeling.ipynb to create and evaluate
models of MLS attendance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from xgboost import XGBRegressor

def create_X(df, columns, add_constant=False, drop_first=True):
    """
    Create a DataFrame to use as input to a model.
    
    Options for columns:
    'home_team',
    'away_team',
    'day',
    'date_month',
    'date_year',
    'local_time',
    'real_home_team',
    'playoff',
    'home_opener',
    'rivals',
    'rain_sum',
    'snow_sum',
    'temperature',
    'windspeed'
    
    Certain columns will be turned into categorical variables.
    All of that work is automatically handled below.
    
    Options:
    add_constant: If True, a constant column will be added. Default is False.
    drop_first: Whether to drop the first column when using get_dummies. Default is True.
    """
    # Check to make sure all entries in columns are valid
    for word in columns:
        if word not in df.columns:
            print(f"Error: {word} is not in the DataFrame.")
            return
    
    # Start with DataFrame containing everything that might be needed
    X = df.copy()[columns]
    
    # Create constant column
    if add_constant:
        X['const'] = 1
    
    # Create categorical variables for home team and away team
    if 'home_team' in columns:
        X = pd.get_dummies(X, columns=['home_team'], drop_first=drop_first, dtype=int)
    if 'away_team' in columns:
        X = pd.get_dummies(X, columns=['away_team'], drop_first=drop_first, dtype=int)
        
    # Create categories for day of the week
    if 'day' in columns:
        X = pd.get_dummies(X, columns=['day'], drop_first=drop_first, dtype=int)
        # Drop Saturday
        X.drop(columns='day_Sat', inplace=True)
    
    # Create categories for year
    if 'date_year' in columns:
        X = pd.get_dummies(X, columns=['date_year'], drop_first=drop_first, dtype=int)
        
    # Create column that indicates if it rained before or during the match
    if 'rain' in columns:
        X['rain_yn'] = df[['rain','rain_sum']].apply(lambda x: 0 if (x.rain==0)&(x.rain_sum==0) else 1, axis=1)
        X.drop(columns='rain', inplace=True)
        
    # Create column that indicates if it snowed before or during the match
    if 'snow' in columns:
        X['snow_yn'] = df[['snow','snow_sum']].apply(lambda x: 0 if (x.snow==0)&(x.snow_sum==0) else 1, axis=1)
        X.drop(columns='snow', inplace=True)

    # Create categories for temperature
    if 'temperature' in columns:
        # The temperature is considered cold if below 40
        X['cold'] = X['temperature'].apply(lambda x: 1 if x<40 else 0)
        # The temperature is considered hot if above 90
        X['hot'] = X['temperature'].apply(lambda x: 1 if x>90 else 0)
        X.drop(columns='temperature', inplace=True)
        
    # Create categories for kick off time
    if 'local_time' in columns:
        X['time_categ'] = X['local_time'].apply(lambda x: 1 if x >= 20 else 0) + \
        X['local_time'].apply(lambda x: 1 if x >= 17 else 0) + \
        X['local_time'].apply(lambda x: 1 if x >= 14 else 0)
        
        X.drop(columns='local_time', inplace=True)
        X = pd.get_dummies(X, columns=['time_categ'], drop_first=drop_first, dtype=int)
        
    # Subtract 7 from the month so that the reference month is July (middle of season)
    if 'date_month' in columns:
        X['date_month'] = X['date_month'].apply(lambda x: x-7)
        
    # Any other columns listed are already in the correct form for the model
        
    return X

def evaluate_linear_model(model, X, y):
    """
    This function takes the results of a model and calculates the following metrics:
    RMSE of the residuals
    The central 68.3% interval of the residuals.
    The R-squared value.
    
    It also makes a plot of predicted vs. actual attendance.
    Inputs:
    model: A fitted linear regression model.
    X: DataFrame that matches formatting and processing of data used to train the model.
    y: Pandas Series, list, or array of target variable values.
    """
    # RMSE of residuals
    print(f"RMSE of residuals: {np.sqrt(model.mse_resid)}")

    # Central 68.3% interval
    print(f"68.3% interval: {np.quantile(model.resid, 0.1585)} {np.quantile(model.resid, 0.8415)}")
    print(f"Width of 68.3% interval: {np.quantile(model.resid, 0.8415)-np.quantile(model.resid, 0.1585)}")
    
    # R-squared value
    print(f"R-squared: {model.rsquared}")
    
    fig, ax = plt.subplots(figsize=(5,5))

    ax.scatter(model.predict(X),
               y,
               alpha=0.5,
               s=7)

    ax.plot([0,80000],[0,80000],color='black')

    ax.set_xlabel('Predicted attendance')
    ax.set_ylabel('Actual attendance')
    
def evaluate_model_split(model, X_train, y_train, X_test, y_test):
    """
    This function makes plots of actual attendance vs. predicted attendance
    for both train and test datasets. It also calculates the following for
    each dataset:
    RMSE of residuals
    R-squared of fit
    Width of 68.3% interval of residuals
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print("RMSE:")
    print(f"Train: {np.round(np.std(y_train-y_train_pred),1)}")
    print(f"Test: {np.round(np.std(y_test-y_test_pred),1)}")
    print()
    print("R-squared:")
    print(f"Train: {np.round(1.0 - np.var(y_train-y_train_pred)/np.var(y_train),3)}")
    print(f"Test: {np.round(1.0 - np.var(y_test-y_test_pred)/np.var(y_test),3)}")
    print()
    print("Width of 68.3% interval")
    print(f"Train: {np.round(np.quantile(y_train-y_train_pred,0.8415)-np.quantile(y_train-y_train_pred,0.1585),1)}")
    print(f"Test: {np.round(np.quantile(y_test-y_test_pred,0.8415)-np.quantile(y_test-y_test_pred,0.1585),1)}")
    
    fig, ax = plt.subplots(ncols=2, figsize=(12,6))
    
    # Train
    ax[0].scatter(y_train_pred,
                  y_train,
                  s=7,
                  alpha=0.5)
    # Plot x=y line
    ax[0].plot([0,80000],[0,80000], color='black')
    ax[0].set_xlabel('Predicted Attendance')
    ax[0].set_xlabel('Actual Attendance')
    ax[0].set_title('Train')
    
    # Test
    ax[1].scatter(y_test_pred,
                  y_test,
                  s=7,
                  alpha=0.5)
    # Plot x=y line
    ax[1].plot([0,80000],[0,80000], color='black')
    ax[1].set_xlabel('Predicted Attendance')
    ax[1].set_xlabel('Actual Attendance')
    ax[1].set_title('Test')
    
    fig.tight_layout()

def xgb_gridsearch(param_grid, X_train, y_train, X_test, y_test):
    """
    This function finds the best combination of hyperparameters for the XGBRegressor
    by training with one dataset, then evaluating using another.
    
    The function evaluates the fit using the R-squared metric, which is the default
    for the XGBRegressor.
    
    Inputs:
    param_grid: Grid of hyperparameters for the XGBRegressor.
    X_train, y_train: Training data
    X_test, y_test: Test data
    
    Outputs:
    xgb_best: Model that achieves the best performance on the test data.
    best_params: Dictionary of the best hyperparameters.
    param_indices: Numpy array of indices for each combination of hyperparameters.
    all_scores: Numpy array of all the train and test scores (R-squared values).
    """
    # Length for each hyperparameter list
    grid_lens = [len(param_grid[key]) for key in param_grid.keys()]
    grid_prods = np.cumprod([1]+grid_lens)
    
    # Initialize the best R-squared to be as bad as possible
    best_score = -1.0
    
    # Best parameters
    best_params = []
    
    # List of all the scores
    all_scores = []
    
    # Parameter indices per model
    param_indices = []
    
    for number in range(grid_prods[-1]):
        # Get the indices for the next combination of parameters
        indices = [(number//grid_prods[x])%grid_lens[x] for x in range(len(grid_lens))]
        
        param_indices.append(indices)
        
        # Make parameter dictionary
        param_dict = {key: param_grid[key][indices[i]] for i,key in enumerate(param_grid.keys())}
        
        # Make new instance of XGBRegressor
        xgb = XGBRegressor(**param_dict)
        
        # Fit to train data
        xgb.fit(X_train, y_train)
        
        # Get test score
        train_score = xgb.score(X_train, y_train)
        test_score = xgb.score(X_test, y_test)
        all_scores.append([train_score, test_score])
        
        # Save if the best model so far
        if test_score > best_score:
            best_score = test_score
            xgb_best = xgb
            best_params = param_dict
            
    
            
    return xgb_best, best_params, np.array(param_indices), np.array(all_scores)