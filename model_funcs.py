"""
The functions in this file are used in modeling.ipynb to create and evaluate
models of MLS attendance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

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
        X = pd.get_dummies(X, columns=['day'], drop_first=False, dtype=int)
        # Drop Saturday
        if drop_first:
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

def model_gridsearch(model_type, param_grid, X_train, y_train, X_test, y_test):
    """
    This function finds the best combination of hyperparameters for the XGBRegressor
    by training with one dataset, then evaluating using another.
    
    The function evaluates the fit using the R-squared metric, which is the default
    for the XGBRegressor.
    
    Inputs:
    model_type: Options: 
    'xgb' for XGBRegressor
    'rfr' for RandomForestRegressor
    'knn' for KNeighborsRegressor
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
        
        # Make new instance of model
        if model_type == 'xgb':
            model = XGBRegressor(**param_dict)
        elif model_type == 'rfr':
            model = RandomForestRegressor(**param_dict)
        elif model_type == 'knn':
            model = KNeighborsRegressor(**param_dict)
        else:
            raise ValueError('Error defining model. Options are xgb, rfr, and knn.')
        
        # Fit to train data
        model.fit(X_train, y_train)
        
        # Get test score
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        all_scores.append([train_score, test_score])
        
        # Save if the best model so far
        if test_score > best_score:
            best_score = test_score
            model_best = model
            best_params = param_dict
            
    return model_best, best_params, np.array(param_indices), np.array(all_scores)

def gen_fake_data(X):
    """
    This function generates fake data that contains all of the features
    present in the input DataFrame, X.
    
    Input: DataFrame that was used to train a model.
    
    Output: A DataFrame of fake matches that contains the same columns as X.
    """
    # Create DataFrame in the same style as the raw data
    fake_df = pd.DataFrame({'day':'Sat',
                            'local_time':19.5,
                            'home_team':[x%29 for x in range(29*29)],
                            'away_team':[x//29 for x in range(29*29)],
                            'playoff':0,
                            'real_home_team':1,
                            'rivals':0,
                            'temperature':70,
                            'rain':0,
                            'snow':0,
                            'windspeed':11,
                            'rain_sum':0,
                            'snow_sum':0,
                            'date_year':2018,
                            'date_month':7,
                            'home_opener':0})
    
    # Drop matches in which a team plays itself
    fake_df.drop(index=fake_df[fake_df['home_team']==fake_df['away_team']].index, inplace=True)
    
    # Use create_X
    X_fake = create_X(fake_df,
                      columns=['home_team','away_team',
                               'day','date_month','date_year','local_time',
                               'playoff','home_opener','rivals','real_home_team',
                               'temperature','rain','snow','windspeed'],
                      drop_first=False)
    
    # Make sure that X_fake contains the same columns as X
    for col in X.columns:
        # If a column is missing, add it
        if col not in X_fake.columns:
            X_fake[col] = 0
            
    # Only keep columns that are in X
    X_fake = X_fake[X.columns]
    
    return X_fake

def get_feature_importance(model, X_fake, split_home_team=False):
    """
    This function produces measures of feature importance for a
    model of MLS attendance. It does this by making predictions
    using fake input data and seeing how those predictions change
    as the features are changed. This allows us to see how important
    each feature is measured in actual attendance.
    
    Inputs:
    model: The model whose features are being tested.
    X_fake: Fake data used for comparing the different features.
    split_home_team: Option for outputting the importance of each feature
    per home team in addition to overall. Default is False.
    
    Output: Dictionary summarizing the importance of each feature.
    If split_home_team is true, it will output a DataFrame with features
    as columns and each row is a home team.
    """
    # Get names of features that were included
    feats = model.feature_names_in_
    
    # Make lists for features and importance
    features = []
    importance = []
    if split_home_team:
        features_ht = []
        importance_ht = []
    
    # Get predictions from X_fake
    y_pred = model.predict(X_fake)
    
    # Get mean attendance of all matches in X_fake
    reference_att = np.mean(y_pred)
    # Get mean attendance per team
    if split_home_team:
        reference_att_ht = [np.mean(y_pred[X_fake[f"home_team_{i}"]==1]) for i in range(29)]
    
    # Home team
    if 'home_team_0' in feats:
        features = features + [f"home_team_{i}" for i in range(29)]
        importance = importance + [np.mean(y_pred[X_fake[f"home_team_{i}"]==1])-reference_att for i in range(29)]
        
    # Away team
    if 'away_team_0' in feats:
        features = features + [f"away_team_{i}" for i in range(29)]
        importance = importance + [np.mean(y_pred[X_fake[f"away_team_{i}"]==1])-reference_att for i in range(29)]
        
    # Day of the week
    if 'day_Mon' in feats:
        # Loop through each day
        for day in ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']:
            # Make new fake data with all days set to day
            X_fake_2 = X_fake.copy()
            X_fake_2['day_Sat'] = 0
            X_fake_2[f"day_{day}"] = 1
            
            # Get predictions
            y_pred_2 = model.predict(X_fake_2)
            features = features + [f"day_{day}"]
            importance = importance + [np.mean(y_pred_2)-reference_att]
            
            if split_home_team:
                features_ht = features_ht + [f"day_{day}"]
                importance_ht.append([np.mean(y_pred_2[X_fake_2[f"home_team_{i}"]==1])-reference_att_ht[i] \
                                      for i in range(29)])
            
    # Month
    if 'date_month' in feats:
        # Loop through each month
        for month in np.arange(2,13):
            # Make new fake data with new month
            X_fake_2 = X_fake.copy()
            X_fake_2['date_month'] = month - 7 # The -7 means 0 is July
            
            # Get predictions
            y_pred_2 = model.predict(X_fake_2)
            features = features + [f"date_month_{month}"]
            importance = importance + [np.mean(y_pred_2)-reference_att]

            if split_home_team:
                features_ht = features_ht + [f"date_month_{month}"]
                importance_ht.append([np.mean(y_pred_2[X_fake_2[f"home_team_{i}"]==1])-reference_att_ht[i] \
                                      for i in range(29)])
                
    # Kick off time
    if 'time_categ_0' in feats:
        # Loop through categories of kick off time
        for cat in range(4):
            # Make new fake data with new time category
            X_fake_2 = X_fake.copy()
            X_fake_2['time_categ_2'] = 0
            X_fake_2[f"time_categ_{cat}"] = 1
            
            # Get predictions
            y_pred_2 = model.predict(X_fake_2)
            features = features + [f"time_categ_{cat}"]
            importance = importance + [np.mean(y_pred_2)-reference_att]
            
            if split_home_team:
                features_ht = features_ht + [f"time_categ_{cat}"]
                importance_ht.append([np.mean(y_pred_2[X_fake_2[f"home_team_{i}"]==1])-reference_att_ht[i] \
                                      for i in range(29)])
            
    # Year
    if 'date_year_2018' in feats:
        # Loop through each year
        for year in [2018,2019,2021,2022,2023]:
            # Make new fake data with new year
            X_fake_2 = X_fake.copy()
            X_fake_2['date_year_2018'] = 0
            X_fake_2[f"date_year_{year}"] = 1
            
            # Get predictions
            y_pred_2 = model.predict(X_fake_2)
            features = features + [f"date_year_{year}"]
            importance = importance + [np.mean(y_pred_2)-reference_att]
            
            if split_home_team:
                features_ht = features_ht + [f"date_year_{year}"]
                importance_ht.append([np.mean(y_pred_2[X_fake_2[f"home_team_{i}"]==1])-reference_att_ht[i] \
                                      for i in range(29)])
            
    # Playoff
    if 'playoff' in feats:
        # Make new fake data where all matches are playoff matches
        X_fake_2 = X_fake.copy()
        X_fake_2['playoff'] = 1
        
        # Get predictions
        y_pred_2 = model.predict(X_fake_2)
        features = features + ['playoff']
        importance = importance + [np.mean(y_pred_2)-reference_att]
        
        if split_home_team:
                features_ht = features_ht + ['playoff']
                importance_ht.append([np.mean(y_pred_2[X_fake_2[f"home_team_{i}"]==1])-reference_att_ht[i] \
                                      for i in range(29)])
        
    # Home Opener
    if 'home_opener' in feats:
        # Make new fake data where all matches are home openers
        X_fake_2 = X_fake.copy()
        X_fake_2['home_opener'] = 1
        
        # Get predictions
        y_pred_2 = model.predict(X_fake_2)
        features = features + ['home_opener']
        importance = importance + [np.mean(y_pred_2)-reference_att]
        
        if split_home_team:
                features_ht = features_ht + ['home_opener']
                importance_ht.append([np.mean(y_pred_2[X_fake_2[f"home_team_{i}"]==1])-reference_att_ht[i] \
                                      for i in range(29)])
        
    # Rivals
    if 'rivals' in feats:
        # Make new fake data where the teams are treated as rivals for each match
        X_fake_2 = X_fake.copy()
        X_fake_2['rivals'] = 1
        
        # Get predictions
        y_pred_2 = model.predict(X_fake_2)
        features = features + ['rivals']
        importance = importance + [np.mean(y_pred_2)-reference_att]
        
        if split_home_team:
                features_ht = features_ht + ['rivals']
                importance_ht.append([np.mean(y_pred_2[X_fake_2[f"home_team_{i}"]==1])-reference_att_ht[i] \
                                      for i in range(29)])
        
    # Real home team
    if 'real_home_team' in feats:
        # Make new fake data where none of the matches have a real home team
        X_fake_2 = X_fake.copy()
        X_fake_2['real_home_team'] = 0
        
        # Get predictions
        y_pred_2 = model.predict(X_fake_2)
        features = features + ['real_home_team']
        importance = importance + [np.mean(y_pred_2)-reference_att]
        
        if split_home_team:
                features_ht = features_ht + ['real_home_team']
                importance_ht.append([np.mean(y_pred_2[X_fake_2[f"home_team_{i}"]==1])-reference_att_ht[i] \
                                      for i in range(29)])
        
    # Rain
    if 'rain_yn' in feats:
        # Make new fake data where it rained for each match
        X_fake_2 = X_fake.copy()
        X_fake_2['rain_yn'] = 1
        
        # Get predictions
        y_pred_2 = model.predict(X_fake_2)
        features = features + ['rain_yn']
        importance = importance + [np.mean(y_pred_2)-reference_att]
        
        if split_home_team:
                features_ht = features_ht + ['rain_yn']
                importance_ht.append([np.mean(y_pred_2[X_fake_2[f"home_team_{i}"]==1])-reference_att_ht[i] \
                                      for i in range(29)])
        
    # Snow
    if 'snow_yn' in feats:
        # Make new fake data where it snowed for each match
        X_fake_2 = X_fake.copy()
        X_fake_2['snow_yn'] = 1
        
        # Get predictions
        y_pred_2 = model.predict(X_fake_2)
        features = features + ['snow_yn']
        importance = importance + [np.mean(y_pred_2)-reference_att]
        
        if split_home_team:
                features_ht = features_ht + ['snow_yn']
                importance_ht.append([np.mean(y_pred_2[X_fake_2[f"home_team_{i}"]==1])-reference_att_ht[i] \
                                      for i in range(29)])
        
    # Cold
    if 'cold' in feats:
        # Make new fake data where it was below 40 degrees for each match
        X_fake_2 = X_fake.copy()
        X_fake_2['cold'] = 1
        
        # Get predictions
        y_pred_2 = model.predict(X_fake_2)
        features = features + ['cold']
        importance = importance + [np.mean(y_pred_2)-reference_att]
        
        if split_home_team:
                features_ht = features_ht + ['cold']
                importance_ht.append([np.mean(y_pred_2[X_fake_2[f"home_team_{i}"]==1])-reference_att_ht[i] \
                                      for i in range(29)])
        
    # Hot
    if 'hot' in feats:
        # Make new fake data where it was above 90 degrees for each match
        X_fake_2 = X_fake.copy()
        X_fake_2['hot'] = 1
        
        # Get predictions
        y_pred_2 = model.predict(X_fake_2)
        features = features + ['hot']
        importance = importance + [np.mean(y_pred_2)-reference_att]
        
        if split_home_team:
                features_ht = features_ht + ['hot']
                importance_ht.append([np.mean(y_pred_2[X_fake_2[f"home_team_{i}"]==1])-reference_att_ht[i] \
                                      for i in range(29)])
                
    if split_home_team:
        # Make DataFrame for importance split by home team
        df_split_ht = pd.DataFrame(data={features_ht[i]:importance_ht[i] for i in range(len(features_ht))})
        
        return {key:val for key, val in zip(features, importance)}, df_split_ht
    else:
        return {key:val for key, val in zip(features, importance)}