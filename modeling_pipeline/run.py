'''
run.py
Author: Mihir Bhaskar
Purpose: run the whole modeling pipeline, including logging results to wandb
Notes:
- This script is designed to be run from the command line, with several arguments parsed through argparse
- Assumes a wandb connection is set up, with user login access to the relevant project (wandb login in the commadn line)
'''
import wandb
from wandb.sklearn import plot_residuals, plot_feature_importances
import numpy as np
import os
import sys
from sqlalchemy import create_engine
import argparse
import pandas as pd

from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

import seaborn as sns
import matplotlib.pyplot as plt

import prep_data as utils
import random

def shuffle_data(X, y, seed):
    '''
    desc: 
        - function to shuffle data, in case we want to selectively train on increasing subsets
    args:
        - X, y: X and y numpy arrays to be shuffled
        - seed (optional): random seed to be used for shuffling the data
    outputs:
        - X and y numpy arrays in new shuffled order
    '''

    # Get list of all row indices
    ind_list = list(range(len(X)))
    
    # Shuffle the list
    random.seed(seed)
    random.shuffle(ind_list)
    
    # Reorder the datasets based on shuffled indices list
    X  = X[ind_list]
    y = y[ind_list]
    
    return X, y

def train(modeltype, X_train, y_train, seed,
          n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
    '''
    args:
        - modeltype: a string of the model class to train. E.g., 'random_forest', 'linear_regression'. See the command line
                     argument parser for a full list of options
        - all types of hyperparameters across model types (e.g., n_estimators, regularization strength, etc.)
        - X_train, y_train: numpy arrays of X and y matrices to train
        - seed: random seed for models that have a random_state
    outputs:
        - fitted sklearn model object, trained on X and y train
    '''
    
    # TODO: find a better way to deal with these hyperparams from command line
    # Maybe parse as a dict? But would anyways need to pull out separate values for wandb logging
    
    # Process hyperparams to make ready for sklearn (if needed)
    if max_features == 'all':
        max_features = None # this is sklearn option for use all features
    
    if modeltype=='random_forest':    
        reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                    max_features=max_features, random_state=seed, n_jobs=-2)
        
    elif modeltype=='linear_regression':
        reg = LinearRegression()
        
    elif modeltype=='poisson':
        reg = PoissonRegressor()
            
    reg.fit(X_train, y_train)
    
    return reg

def predict(reg, X):
    '''
    args:
        - reg: a fitted sklearn model object
        - X: x matrix for which predictions for target to be created
    outputs:
        - numpy array of predicted y values
    '''
    
    return reg.predict(X)
    
def run_experiment(modeltype, n, full_data_used, X_train, X_test, y_train, y_test, num_cv_splits, seed,
                   n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
    '''
    desc: stitches the whole training-prediction pipeline together as an 'experiment'
    
    args:
        - modeltype: a string of the model class (see argument parser for options)
        - n: max number of observations to train on 
        - full_data_used: boolean to indicate whether full training data used (i.e., n = len(X_train))
        - train and test X and y matrices as numpy arrays
        - num_cv_splits: number of splits to make in training for cross-validation
        - seed: random seed
        - hyperparameters across all models (e.g., n_estimators, max_depth, etc.)
    outputs:
        - logs several metrics and outputs to a wandb run on the server.
          The config in wandb init is the key metadata about the run,
          and wandb.log is all the info that is associated with the run
    '''
        
    # TODO: think about some if condition here for logging only the relevant hyperparams in the config
    # and setting the rest to none. Right now, all hyperparams will log for all model types (using the default values)
    # even if it doesn't make sense. Will require the user, when analysing results, to focus on the relevant columns
    # for each model class.
    
    wandb.init(project='eruka-housing', entity='gormleylab',
               name=f'{modeltype}_{n}',
               config={'modeltype': modeltype,
                       'n': n,
                       'full_data_used': full_data_used,
                       'num_cv_splits': num_cv_splits,
                       'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'max_features': max_features})
        
    # Note: here, I pull the hyperparameters from the wandb init config (rather than directly using the command line variables)
    # This is to allow for wandb sweeps to be configured for hyperparamter search (where we loop across different config vals)
        
    model = train(modeltype, X_train, y_train, seed,
                   n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features)
    
    y_pred = predict(model, X_test)
    y_train_pred = predict(model, X_train)
    
    # Calculating cross-validated metrics
    cross_val_mses = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error',
                                     cv=num_cv_splits, n_jobs=-2)
    cross_val_r2 = cross_val_score(model, X_train, y_train, scoring='r2', cv=num_cv_splits, n_jobs=-2)
    
    
    # Metrics to log
    cv_rmse = np.mean(np.sqrt(np.abs(cross_val_mses)))
    cv_r2 = np.mean(cross_val_r2)
    test_rmse = mean_squared_error(y_test, y_pred, squared=False)
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    test_r2 = r2_score(y_test, y_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    wandb.log({'test_rmse': test_rmse, 
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'cv_rmse': cv_rmse,
            'cv_r2': cv_r2
            })
    
    # Plots to log
    
    # wandb.sklearn.plot_regressor(model, X_train, X_test, y_train, y_test, modeltype) # Note: this plot_regressor methods takes too much time because it creates unnecessary graphs
    
    # Log residuals plot for test data
    wandb.sklearn.plot_residuals(model, X_test, y_test)
    
    # Log feature importance if the method allows for it    
    wandb.sklearn.plot_feature_importances(model)

    # TODO plots:
    # Real vs. predicted value, colored by different characteristics
    # Deep dive into understanding who the model
        
    wandb.finish()
    

if __name__ == '__main__':
    
    # Arguments to be entered through command line: see 'help' for description
    
    # Data processing-related arugments 
    parser = argparse.ArgumentParser(description="Script to run modeling pipeline")
    parser.add_argument('-regen', '--regen_matrices', action='store_true', required=False, help='rerun creation of train-test matrices or use previously stored')
    parser.add_argument('-sh', '--shuffle', action='store_true', required=False, help='shuffle training data')
    parser.add_argument('--n', type=int, action='store', default=100000, required=False, help="max number of rows to use in train data")
    parser.add_argument('--cvsplits', type=int, action='store', default=4, required=False, help="number of splits for cross-validation metrics")
    parser.add_argument('--testprop', type=float, action='store', default=0.2, required=False, help="proportion of data to hold out for test")

    # Model-related arguments
    parser.add_argument('modeltype', action='store', choices = ['random_forest', 'linear_regression', 'poisson'], help="name of model type to run") # the only required positional argument
    
    parser.add_argument('--n_estimators', type=int, action='store', default=100, required=False, help="number of trees in forest (tree models)")
    parser.add_argument('--max_depth', type=int, action='store', default=10000000, required=False, help="max depth of tree (tree models)")
    parser.add_argument('--min_samples_split', type=int, action='store', default=2, required=False, help="min samples required to split at node (tree models)")
    parser.add_argument('--min_samples_leaf', type=int, action='store', default=1, required=False, help="min samples required at leaf node (tree models)")
    parser.add_argument('--max_features', action='store', choices=['sqrt', 'log2', 'all'], default='sqrt', required=False, help="max features to consider when splitting (tree models)")

    # Miscellaneous arguments
    parser.add_argument('-seed', type=int, action='store', default=12345, required=False, help="random seed to be used for all random processes")

    args = parser.parse_args()

    # Set DB connection from environment
    if 'ERUKA_DB' not in os.environ or not os.environ['ERUKA_DB']:
        print('No PostgreSQL endpoing configured, please specify connection string via ERUKA_DB environment variable')
        sys.exit()
        
    db_uri = os.environ['ERUKA_DB']
    db_engine = create_engine(db_uri)
    
    # Get chosen seed from command line (or default if nothing entered)
    seed = args.seed
    
    # Either load data or recreate matrices (based on command line flag)
    if not args.regen_matrices:
        X_train = np.genfromtxt('matrices/X_train.txt')
        X_test = np.genfromtxt('matrices/X_test.txt')
        y_train = np.genfromtxt('matrices/y_train.txt')
        y_test = np.genfromtxt('matrices/y_test.txt')
        
        # Reading column names into colnames
        with open('matrices/colnames.txt', 'r') as file:
            colnames = [line for line in file]
            
    else:
        testprop = args.testprop # proportion of 
        X_train, X_test, y_train, y_test, colnames = utils.main(db_engine, keep='simple', test_size=testprop, 
                                                      random_state=seed, matrix_path='matrices')
    
    # Shuffle data if desired
    if args.shuffle:
        X_train, y_train = shuffle_data(X_train, y_train, seed=5555)
        
    # Subset to max number of training observations
    full_data_used = True
    if args.n < len(X_train):
        full_data_used = False
    
    n = min(args.n, len(X_train)) # if user entered more than length of training data, then all training observations used
    X_train = X_train[:n, :]
    y_train = y_train[:n]
    
    # Get model type and hyperparameters
    modeltype = args.modeltype
    
    n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features = args.n_estimators, args.max_depth, args.min_samples_split, args.min_samples_leaf, args.max_features
    
    num_cv_splits = args.cvsplits
    
    # Print some important outputs as sanity check
    print(f"\nShape of X_train = {X_train.shape}, shape of y_train = {y_train.shape}\n")
        
    run_experiment(modeltype, n, full_data_used, X_train, X_test, y_train, y_test, num_cv_splits, seed,
                   n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features)    
    
