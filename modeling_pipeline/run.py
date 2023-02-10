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

import seaborn as sns
import matplotlib.pyplot as plt

import prep_data as utils

import random



# def sample_data(df, n):
#     return df.sample(n=n, random_state=4)

def train(X_train, y_train):
    reg = RandomForestRegressor(random_state=4)
    reg.fit(X_train, y_train)
    return reg

def predict(reg, X_test):
    return reg.predict(X_test)
    
def run_experiment(n, X_train, X_test, y_train, y_test):
    
    wandb.init(project='eruka-housing', entity='gormleylab',
               name=f'testrun_reg_{n}',
               config={'modeltype': 'random forest',
                       'n': n})
        
    model = train(X_train, y_train)
    
    y_pred = predict(model, X_test)
    
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    wandb.log({'rmse': rmse, 'r2': r2})
    
   # wandb.sklearn.plot_residuals(model, X_train, y_train)
        
    wandb.finish()
    

if __name__ == '__main__':
    
    # # Arguments to be entered through command line
    # parser = argparse.ArgumentParser(description="Script to run modeling pipeline")
    # parser.add_argument('regen_matrix', type=)
    # parser.add_argument('')
    # parser.add_argument('n', type=int, help='Number of labeled points to user from the data')
    # parser.add_argument('-t', '--types', action='store_true', choices=['simple', 'all'], default='simple',
    #                                                         help='Whether to keep simple case where no year and not handwritte, or all')
    # args = parser.parse_args()

    
    # Argparse options to add:
    # whether to rerun prep_data or just pull from matrices folder
    # Whether to shuffle + pick n of the data, or use all
    # 
    
    # To change directly in file:
        # Model + hyperparams
        # 
    
    # Change the config in run.py itself to run a certain hyperparam
    
    
    # Add argparse option to regen matrices (basically whether to run prep_data or no)
    # If no, just pull stuff from the matrices folder
    
    # Set DB connection from environment
    if 'ERUKA_DB' not in os.environ or not os.environ['ERUKA_DB']:
        print('No PostgreSQL endpoing configured, please specify connection string via ERUKA_DB environment variable')
        sys.exit()
        
    db_uri = os.environ['ERUKA_DB']
    db_engine = create_engine(db_uri)
    
    # Get parameter for sample size
    # n = args.n
    
    # Read full data    
    #X_train, X_test, y_train, y_test = utils.main(db_engine, keep='simple', test_size=0.2, random_state=4, matrix_path='matrices')
    X_train = np.genfromtxt('matrices/X_train.txt')
    X_test = np.genfromtxt('matrices/X_test.txt')
    y_train = np.genfromtxt('matrices/y_train.txt')
    y_test = np.genfromtxt('matrices/y_test.txt')
    
    # Shuffle data
    ind_list = list(range(len(X_train)))
    random.seed(12345)
    random.shuffle(ind_list)
    X_train_shuffled  = X_train[ind_list]
    y_train_shuffled = y_train[ind_list]
        
    # Sample increasing amounts
    n_list = [500, 1000, 1500, 2000, 2500, 3000, 3500]
    for n in n_list:
        X_train_sub = X_train_shuffled[:n, :]
        y_train_sub = y_train_shuffled[:n]
        run_experiment(n, X_train_sub, X_test, y_train_sub, y_test)    
    
