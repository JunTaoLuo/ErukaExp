'''
run.py
Author: Mihir Bhaskar
Purpose: run the whole modeling pipeline, including logging results to wandb
'''
import wandb
from wandb.sklearn import plot_precision_recall, plot_feature_importances
import numpy as np
import os
import sys
from sqlalchemy import create_engine
import argparse
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

import seaborn as sns
import matplotlib.pyplot as plt

import prep_data as utils
import random

def shuffle_data(X, y, seed):
    '''
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

def train(modeltype, X_train, y_train, seed):
    
    if modeltype=='random_forest':
        reg = RandomForestRegressor(random_state=seed)
    elif modeltype=='linear_regression':
        reg = LinearRegression()
            
    reg.fit(X_train, y_train)
    return reg

def predict(reg, X_test):
    return reg.predict(X_test)
    
def run_experiment(modeltype, n, X_train, X_test, y_train, y_test, seed):
    
    wandb.init(project='eruka-housing', entity='gormleylab',
               name=f'{modeltype}_{n}',
               config={'modeltype': modeltype,
                       'n': n})
        
    model = train(modeltype, X_train, y_train, seed)
    
    y_pred = predict(model, X_test)
    y_train_pred = predict(model, X_train)
    
    # Calculating cross-validated metrics 5 splits (make splits an argument)
    cross_val_mses = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error',
                                     cv=5, n_jobs=-2)
    cross_val_r2 = cross_val_score(model, X_train, y_train, scoring='r2', cv=5, n_jobs=-2)
    
    
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
    
    
    # Plots to create:
    
    # Real vs. predicted value, colored by different characteristics
    # Deep dive into understanding who the model
    
   # wandb.sklearn.plot_residuals(model, X_train, y_train)
        
    wandb.finish()
    

if __name__ == '__main__':
    
    # Arguments to be entered through command line
    
    # Data processing-related arugments
    parser = argparse.ArgumentParser(description="Script to run modeling pipeline")
    parser.add_argument('-regen', '--regen_matrices', action='store_true', required=False, help='rerun creation of train-test matrices or use previously stored')
    parser.add_argument('-sh', '--shuffle', action='store_true', required=False, help='shuffle training data')
    parser.add_argument('-n', type=int, action='store', default=100000, required=False, help="max number of rows to use in train data")
    
    # Model-related arguments
    parser.add_argument('mtype', action='store', choices = ['random_forest', 'linear_regression'], help="name of model type to run")


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
    else:
        X_train, X_test, y_train, y_test = utils.main(db_engine, keep='simple', test_size=0.2, 
                                                      random_state=seed, matrix_path='matrices')
    
    # Shuffle data if desired
    if args.shuffle:
        X_train, y_train = shuffle_data(X_train, y_train, seed=5555)
        
    # Subset to max number of training observations
    n = min(args.n, len(X_train)) # if user entered more than length of training data, then all training observations used
    X_train = X_train[:n, :]
    y_train = y_train[:n]
    
    # Get model type and hyperparameters
    modeltype = args.mtype
    
    # Print some important outputs
    print(f"\nShape of X_train = {X_train.shape}, shape of y_train = {y_train.shape}\n")
    
    run_experiment(modeltype, n, X_train, X_test, y_train, y_test, seed)    
    
