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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

import seaborn as sns
import matplotlib.pyplot as plt

import prep_data as utils
import random

def get_perc_error(y_pred, y_true):
    '''
    args:
        - y_pred: numpy array of predicted values
        - y_true: numpy array of true label values
    outputs:
        - numpy array of % error for each observation
    '''

    return 100*abs((y_pred - y_true)/y_true)

def get_squared_perc_error(y_pred, y_true):

    return ((y_pred - y_true)/y_true)**2


def plot_true_pred(y_pred, y_true, truncatedView=False):
    '''
    args:
        - y_pred: numpy array of predicted values
        - y_true: numpy array of true label values
    outputs:
        - matplotlib ax object configured to plot y-true against y-pred
    '''
    # TODO: add argument to color the scatter points by another feature value

    f, ax = plt.subplots(figsize=(10,10))
    ax.set_title('Predicted vs. true value')
    ax.scatter(y_true, y_pred, c='crimson')

    # Adding x=y line
    p1 = max(max(y_true), max(y_true))
    p2 = min(min(y_pred), min(y_pred))
    ax.plot([p1, p2], [p1, p2], 'b-')

    ax.set_xlabel('True Values', fontsize=15)
    ax.set_ylabel('Predictions', fontsize=15)

    if truncatedView:
        ax.set_xlim((1000, 7000))
    else:
        ax.axis('equal')

    return ax


def run_experiment(y_test, y_pred):
    '''
    desc: Runs baseline model with hand tuned parameters
    '''

    # TODO: think about some if condition here for logging only the relevant hyperparams in the config
    # and setting the rest to none. Right now, all hyperparams will log for all model types (using the default values)
    # even if it doesn't make sense. Will require the user, when analysing results, to focus on the relevant columns
    # for each model class.

    wandb.init(project='eruka-housing', entity='gormleylab',
               name=f'baseline',
               config={'comments':'baseline estimate'})

    # Note: here, I pull the hyperparameters from the wandb init config (rather than directly using the command line variables)
    # This is to allow for wandb sweeps to be configured for hyperparamter search (where we loop across different config vals)

    # Metrics to log

    # Standard regression stats
    test_rmse = mean_squared_error(y_test, y_pred, squared=False)
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    # Error distance stats (how wrong were we)
    test_perc_error = get_perc_error(y_pred, y_test)
    median_perc_error = np.percentile(test_perc_error, 50)
    within_5_perc_error = 100*(np.mean(test_perc_error <= 5))
    within_10_perc_error = 100*(np.mean(test_perc_error <= 10))
    within_20_perc_error =100*(np.mean(test_perc_error <= 20))

    # Subset
    perc_5 = 1140
    perc_95 = 6715

    test_subset = y_test[(y_test >= perc_5)&(y_test <= perc_95)]
    pred_subset = y_pred[(y_test >= perc_5)&(y_test <= perc_95)]

    print(f"Length of Hamilton full test set: {len(y_test)}")
    print(f"Length of Hamilton 5-95 perc test subset: {len(test_subset)}")

    test_rmse_sub = mean_squared_error(test_subset, pred_subset, squared=False)
    test_r2_sub = r2_score(test_subset, pred_subset)
    test_mae_sub = mean_absolute_error(test_subset, pred_subset)

    # Error distance stats on subset
    test_perc_error_sub = get_perc_error(pred_subset, test_subset)

    mape_sub = np.mean(test_perc_error_sub)

    median_perc_error_sub = np.percentile(test_perc_error_sub, 50)
    within_5_perc_error_sub = 100*(np.mean(test_perc_error_sub <= 5))
    within_10_perc_error_sub = 100*(np.mean(test_perc_error_sub <= 10))
    within_20_perc_error_sub = 100*(np.mean(test_perc_error_sub <= 20))

    squared_perc_error_sub = get_squared_perc_error(pred_subset, test_subset)
    mspe_sub = np.mean(squared_perc_error_sub)
    rmspe_sub = np.sqrt(mspe_sub)

    # Error on subsets
    test_rmse_25perc_lowest = mean_squared_error(y_test[y_test <= 2250], y_pred[y_test <= 2250], squared=False)
    test_rmse_50perc_lowest = mean_squared_error(y_test[y_test <= 3085], y_pred[y_test <= 3085], squared=False)
    test_rmse_75perc_lowest = mean_squared_error(y_test[y_test <= 4110], y_pred[y_test <= 4110], squared=False)
    test_rmse_85perc_lowest = mean_squared_error(y_test[y_test <= 4850], y_pred[y_test <= 4850], squared=False)
    test_rmse_90perc_lowest = mean_squared_error(y_test[y_test <= 5489], y_pred[y_test <= 5489], squared=False)
    test_rmse_95perc_lowest = mean_squared_error(y_test[y_test <= 6714], y_pred[y_test <= 6714], squared=False)

    # Franklin performance, if generalized model selected
    franklin_1920_rmse = 0
    franklin_1931_rmse = 0

    f31_median_perc_error_sub = 0
    f31_within_5_perc_error_sub = 0
    f31_within_10_perc_error_sub = 0
    f31_within_20_perc_error_sub = 0

    # Plots to log

    # Plot true vs predicted value
    true_pred_plot_test = plot_true_pred(y_pred, y_test)
    true_pred_plot_test_sub = plot_true_pred(pred_subset, test_subset, truncatedView=True)
    # plt.show()

    wandb.log({'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'median_test_error_perc': median_perc_error,
            'within_5perc_testerror': within_5_perc_error,
            'within_10perc_testerror': within_10_perc_error,
            'within_20perc_testerror': within_20_perc_error,
            'test_rmse_25perc_lowest': test_rmse_25perc_lowest,
            'test_rmse_50perc_lowest': test_rmse_50perc_lowest,
            'test_rmse_75perc_lowest': test_rmse_75perc_lowest,
            'test_rmse_85perc_lowest': test_rmse_85perc_lowest,
            'test_rmse_90perc_lowest': test_rmse_90perc_lowest,
            'test_rmse_95perc_lowest': test_rmse_95perc_lowest,
            'true_pred_plot_test': wandb.Image(true_pred_plot_test),
            'true_pred_plot_test_sub': wandb.Image(true_pred_plot_test_sub),
            'test_rmse_sub': test_rmse_sub,
            'test_r2_sub': test_r2_sub,
            'test_mae_sub': test_mae_sub,
            'median_perc_error_sub': median_perc_error_sub,
            'mape_sub': mape_sub,
            'mspe_sub': mspe_sub,
            'rmspe_sub': rmspe_sub,
            'within_5_perc_error_sub': within_5_perc_error_sub,
            'within_10_perc_error_sub': within_10_perc_error_sub,
            'within_20_perc_error_sub': within_20_perc_error_sub,
            })

    wandb.finish()


if __name__ == '__main__':

    baseline = pd.read_csv("baseline.csv", names=["parcelid", "labels", "estimates"])
    y_test = baseline["labels"].to_numpy()
    y_pred = baseline["estimates"].to_numpy()

    # Run the experiment
    run_experiment(y_test, y_pred)
