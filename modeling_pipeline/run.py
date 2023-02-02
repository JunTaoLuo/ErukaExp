import wandb
from wandb.sklearn import plot_precision_recall, plot_feature_importances
from wandb.sklearn import plot_class_proportions, plot_learning_curve, plot_roc
import numpy as np
import os
import sys
from sqlalchemy import create_engine
import argparse
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score 

import seaborn as sns
import matplotlib.pyplot as plt

import prep_data as utils

def gen_matrices(engine, keep, test_size):
    df = utils.read_data(engine, keep)
    X_train, X_test, y_train, y_test = utils.split_data(df, test_size)
    X_train = utils.process_features(X_train)
    X_test = utils.process_features(X_test)
    
    return X_train, X_test, y_train, y_test

def train(X_train, y_train):
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    return reg

def predict(reg, X_test):
    return reg.predict(X_test)
    
def run_experiment(engine, keep, test_size):
    
    wandb.init(project='eruka-housing', entity='gormleylab',
               name='testrun_regression',
               config={'modeltypes': 'linear regression'})
    
    X_train, X_test, y_train, y_test = gen_matrices(engine, keep, test_size)
    
    model = train(X_train, y_train)
    
    y_pred = predict(model, X_test)
    
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred, )
    
    wandb.log({'rmse': rmse, 'r2': r2})
    
    wandb.sklearn.plot_residuals(model, X_train, y_train)
    
    # wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test, model_name='Ridge')
    
    wandb.finish()
    

if __name__ == '__main__':
    
    # parser = argparse.ArgumentParser(description="Script to run modeling pipeline")
    # parser.add_argument('-n', '--numsamples', action='store_true', help='Number of labeled points to user from the data')
    # parser.add_argument('-t', '--types', action='store_true', choices=['simple', 'all'], default='simple',
    #                                                         help='Whether to keep simple case where no year and not handwritte, or all')
    # args = parser.parse_args()

    # Set DB connection from environment
    if 'ERUKA_DB' not in os.environ or not os.environ['ERUKA_DB']:
        print('No PostgreSQL endpoing configured, please specify connection string via ERUKA_DB environment variable')
        sys.exit()
        
    db_uri = os.environ['ERUKA_DB']
    db_engine = create_engine(db_uri)
    
    run_experiment(db_engine, keep='simple', test_size=0.2)
    
    
