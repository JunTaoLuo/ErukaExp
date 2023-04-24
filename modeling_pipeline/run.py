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
          n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,
          alpha):
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
        reg = PoissonRegressor(alpha=alpha)

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

def plot_true_pred(y_pred, y_true):
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
    ax.axis('equal')

    return ax


def run_experiment(modeltype, n, trainsource, full_data_used, keep, X_train, X_test, y_train, y_test,
                   franklin, X_franklin_1920, y_franklin_1920, X_franklin_1931, y_franklin_1931,
                   colnames, num_cv_splits, seed,
                   n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,
                   alpha, ocr_threshold, comments=''):
    '''
    desc: stitches the whole training-prediction pipeline together as an 'experiment'

    args:
        - modeltype: a string of the model class (see argument parser for options)
        - n: max number of observations to train on
        - trainsource: whether to train using ocr-only, hand-labeled only or both
        - full_data_used: boolean to indicate whether full training data used (i.e., n = len(X_train))
        - keep: 'all' (all handlabeled points) or
                'simple' (remove nonstandard handlabeled cases where year is entered or card is handwritten)
        - train and test X and y matrices as numpy arrays
        - num_cv_splits: number of splits to make in training for cross-validation
        - seed: random seed
        - hyperparameters across all models (e.g., n_estimators, max_depth, etc.)
        - comments: string to feed in any other model specific info (e.g., some niche thing we want to test out in wandb)
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
                       'franklin': franklin,
                       'n': n,
                       'trainsource': trainsource,
                       'full_data_used': full_data_used,
                       'keep': keep,
                       'num_cv_splits': num_cv_splits,
                       'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'max_features': max_features,
                       'alpha': alpha,
                       'ocr_threshold': ocr_threshold,
                       'comments':comments})

    # Note: here, I pull the hyperparameters from the wandb init config (rather than directly using the command line variables)
    # This is to allow for wandb sweeps to be configured for hyperparamter search (where we loop across different config vals)

    model = train(modeltype, X_train, y_train, seed,
                   n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,
                   alpha)

    y_pred = predict(model, X_test)
    y_train_pred = predict(model, X_train)

    # Calculating cross-validated metrics
    cross_val_mses = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error',
                                     cv=num_cv_splits, n_jobs=-2)
    cross_val_r2 = cross_val_score(model, X_train, y_train, scoring='r2', cv=num_cv_splits, n_jobs=-2)


    # Metrics to log

    # Standard regression stats
    cv_rmse = np.mean(np.sqrt(np.abs(cross_val_mses)))
    cv_r2 = np.mean(cross_val_r2)
    test_rmse = mean_squared_error(y_test, y_pred, squared=False)
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    test_r2 = r2_score(y_test, y_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)

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

    if franklin is True:
        y_pred_franklin_1920 = predict(model, X_franklin_1920)
        y_pred_franklin_1931 = predict(model, X_franklin_1931)

        franklin_1920_rmse = mean_squared_error(y_franklin_1920, y_pred_franklin_1920, squared=False)
        franklin_1931_rmse = mean_squared_error(y_franklin_1931, y_pred_franklin_1931, squared=False)

        # Subset stats for 1931
        f31_sub = y_franklin_1931[(y_franklin_1931 >= perc_5)&(y_franklin_1931 <= perc_95)]
        f31_pred_sub = y_pred_franklin_1931[(y_franklin_1931 >= perc_5)&(y_franklin_1931 <= perc_95)]

        print(f"Length of Franklin full test set: {len(y_franklin_1931)}")
        print(f"Length of Franklin 5-95 perc test subset: {len(f31_sub)}")

        f31_rmse_sub = mean_squared_error(f31_sub, f31_pred_sub, squared=False)
        # Error distance stats on subset
        f31_perc_error_sub = get_perc_error(f31_pred_sub, f31_sub)
        f31_mape_sub = np.mean(f31_perc_error_sub)
        f31_squared_perc_error_sub = get_squared_perc_error(f31_pred_sub, f31_sub)

        f31_mspe_sub = np.mean(f31_squared_perc_error_sub)
        f31_rmspe_sub = np.sqrt(f31_mspe_sub)

        f31_median_perc_error_sub = np.percentile(f31_perc_error_sub, 50)
        f31_within_5_perc_error_sub = 100*(np.mean(f31_perc_error_sub <= 5))
        f31_within_10_perc_error_sub = 100*(np.mean(f31_perc_error_sub <= 10))
        f31_within_20_perc_error_sub = 100*(np.mean(f31_perc_error_sub <= 20))

    # Plots to log

    # Plot true vs predicted value
    true_pred_plot_test = plot_true_pred(y_pred, y_test)
    true_pred_plot_train = plot_true_pred(y_train_pred, y_train)

    true_pred_plot_test_sub = plot_true_pred(pred_subset, test_subset)



    # wandb.sklearn.plot_regressor(model, X_train, X_test, y_train, y_test, modeltype) # Note: this plot_regressor methods takes too much time because it creates unnecessary graphs

    # Log residuals plot for test data
    # wandb.sklearn.plot_residuals(model, X_test, y_test)

    # Log feature importance if the method allows for it
    wandb.sklearn.plot_feature_importances(model)

    if franklin is True:
        wandb.log({'test_rmse': test_rmse,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                'cv_rmse': cv_rmse,
                'cv_r2': cv_r2,
                'train_mae': train_mae,
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
                'true_pred_plot_train': wandb.Image(true_pred_plot_train),
                'true_pred_plot_test_sub': wandb.Image(true_pred_plot_test_sub),
                'franklin_1920_rmse': franklin_1920_rmse,
                'franklin_1931_rmse': franklin_1931_rmse,
                'test_rmse_sub': test_rmse_sub,
                'test_r2_sub': test_r2_sub,
                'test_mae_sub': test_mae_sub,
                'f31_rmse_sub': f31_rmse_sub,
                'median_perc_error_sub': median_perc_error_sub,
                'mape_sub': mape_sub,
                'mspe_sub': mspe_sub,
                'rmspe_sub': rmspe_sub,
                'within_5_perc_error_sub': within_5_perc_error_sub,
                'within_10_perc_error_sub': within_10_perc_error_sub,
                'within_20_perc_error_sub': within_20_perc_error_sub,
                'f31_median_perc_error_sub': f31_median_perc_error_sub,
                'f31_mape_sub': f31_mape_sub,
                'f31_mspe_sub': f31_mspe_sub,
                'f31_rmspe_sub': f31_rmspe_sub,
                'f31_within_5_perc_error_sub': f31_within_5_perc_error_sub,
                'f31_within_10_perc_error_sub': f31_within_10_perc_error_sub,
                'f31_within_20_perc_error_sub': f31_within_20_perc_error_sub,
                })
    else:
        wandb.log({'test_rmse': test_rmse,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                'cv_rmse': cv_rmse,
                'cv_r2': cv_r2,
                'train_mae': train_mae,
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
                'true_pred_plot_train': wandb.Image(true_pred_plot_train),
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

    # Arguments to be entered through command line: see 'help' for description

    # Data processing-related arugments
    parser = argparse.ArgumentParser(description="Script to run modeling pipeline")
    parser.add_argument('-regen', '--regen_matrices', action='store_true', required=False, help='rerun creation of train-test matrices or use previously stored')
    parser.add_argument('-sh', '--shuffle', action='store_true', required=False, help='shuffle training data')
    parser.add_argument('--n', type=int, action='store', default=100000, required=False, help="max number of rows to use in train data")
    parser.add_argument('--cvsplits', type=int, action='store', default=4, required=False, help="number of splits for cross-validation metrics")
    parser.add_argument('--testprop', type=float, action='store', default=0.2, required=False, help="proportion of data to hold out for test")
    parser.add_argument('--trainsource', action='store', choices=['ocr', 'hand', 'both'], default='hand', required=False, help="whether to use hand-labeled data, ocr data, or both for training")
    parser.add_argument('--testsource', action='store', choices=['train', 'segmentation_errors'], default='train', required=False, help="whether to use trainsource partition or another source for testing")
    parser.add_argument('--keep', action='store', choices=['simple', 'all'], default='simple', required=False, help='''whether to use all hand-labeled data including handwritten/year, with appropriate flags as features,
                                                                                                                    or just the simple hand-labeled cases where year is null and handwritten is null''')

    # OCR related parameters
    parser.add_argument('--ocrthreshold', type=float, action='store', default=-9999, required=False, help="OCR Threshold")
    parser.add_argument('--ocrsource', action='store', required=False, help="Path to the ocr results")
    parser.add_argument('--skiplargen', action='store_true', required=False, help="Skip run when n > number of training samples")

    # Model-related arguments
    parser.add_argument('modeltype', action='store', choices = ['random_forest', 'linear_regression', 'poisson'], help="name of model type to run") # the only required positional argument

    parser.add_argument('--franklin', action='store_true', required=False, help='whether to train generalizable model and test on franklin')

    parser.add_argument('--n_estimators', type=int, action='store', default=100, required=False, help="number of trees in forest (tree models)")
    parser.add_argument('--max_depth', type=int, action='store', default=10000000, required=False, help="max depth of tree (tree models)")
    parser.add_argument('--min_samples_split', type=int, action='store', default=2, required=False, help="min samples required to split at node (tree models)")
    parser.add_argument('--min_samples_leaf', type=int, action='store', default=1, required=False, help="min samples required at leaf node (tree models)")
    parser.add_argument('--max_features', action='store', choices=['sqrt', 'log2', 'all'], default='sqrt', required=False, help="max features to consider when splitting (tree models)")

    parser.add_argument('--alpha', action='store', type=float, default=1.0, required=False, help="regularization strength (linear models), 0=no regularization")

    # Miscellaneous arguments
    parser.add_argument('--seed', type=int, action='store', default=12345, required=False, help="random seed to be used for all random processes")
    parser.add_argument('--comments', action='store', default='', required=False, help="Any comments on the model")


    args = parser.parse_args()

    # Set DB connection from environment
    if 'ERUKA_DB' not in os.environ or not os.environ['ERUKA_DB']:
        print('No PostgreSQL endpoing configured, please specify connection string via ERUKA_DB environment variable')
        sys.exit()

    db_uri = os.environ['ERUKA_DB']
    db_engine = create_engine(db_uri)

    # Get chosen seed from command line (or default if nothing entered)
    seed = args.seed

    # TODO: make code below more efficient by directly reading in relevant matrix based on trainsource type

    # Either load data or recreate matrices (based on command line flag)

    keep = args.keep

    if not args.regen_matrices:
        X_train_hand = np.genfromtxt('matrices/X_train_hand.txt')
        X_train_ocr = np.genfromtxt('matrices/X_train_ocr.txt')
        X_test = np.genfromtxt('matrices/X_test.txt')
        X_train_hand_sub = np.genfromtxt('matrices/X_train_hand_sub.txt')
        X_train_ocr_sub = np.genfromtxt('matrices/X_train_ocr_sub.txt')
        X_test_sub = np.genfromtxt('matrices/X_test_sub.txt')
        y_train_hand = np.genfromtxt('matrices/y_train_hand.txt')
        y_train_ocr = np.genfromtxt('matrices/y_train_ocr.txt')
        y_test = np.genfromtxt('matrices/y_test.txt')
        X_franklin_1920 = np.genfromtxt('matrices/X_franklin_1920.txt')
        X_franklin_1931 = np.genfromtxt('matrices/X_franklin_1931.txt')
        y_franklin_1920 = np.genfromtxt('matrices/y_franklin_1920.txt')
        y_franklin_1931 = np.genfromtxt('matrices/y_franklin_1931.txt')
        X_test_segmentation_error = np.genfromtxt('matrices/X_test_segmentation_error.txt')
        y_test_segmentation_error = np.genfromtxt('matrices/y_test_segmentation_error.txt')

        # Reading column names into colnames
        with open('matrices/colnames.txt', 'r') as file:
            colnames = [line for line in file]
        with open('matrices/colnames_sub.txt') as file:
            colnames_sub = [line for line in file]

    else:
        testprop = args.testprop # proportion of observations to keep in test set
        X_train_hand, X_train_ocr, X_test, X_train_hand_sub, X_train_ocr_sub, X_test_sub, y_train_hand, y_train_ocr, y_test, X_franklin_1920, X_franklin_1931, y_franklin_1920, y_franklin_1931, X_test_segmentation_error, y_test_segmentation_error, colnames, colnames_sub = utils.main(db_engine,
                                                                                                    ocr_threshold=args.ocrthreshold,
                                                                                                    keep=keep,
                                                                                                    ocr_path=args.ocrsource,
                                                                                                    test_size=testprop,
                                                                                                    random_state=seed,
                                                                                                    matrix_path='matrices')

    # Create desired X_train based on OCR-only, hand-labeled only, or both and generalized model/full model
    if not args.franklin:
        if args.trainsource == 'hand':
            X_train = X_train_hand
            y_train = y_train_hand
        elif args.trainsource == 'ocr':
            X_train = X_train_ocr
            y_train = y_train_ocr
        elif args.trainsource == 'both':
            X_train = np.concatenate((X_train_hand, X_train_ocr), axis=0)
            y_train = np.concatenate((y_train_hand, y_train_ocr), axis=0)
    if args.franklin:
        if args.trainsource == 'hand':
            X_train = X_train_hand_sub
            y_train = y_train_hand
        elif args.trainsource == 'ocr':
            X_train = X_train_ocr_sub
            y_train = y_train_ocr
        elif args.trainsource == 'both':
            X_train = np.concatenate((X_train_hand_sub, X_train_ocr_sub), axis=0)
            y_train = np.concatenate((y_train_hand, y_train_ocr), axis=0)
        X_test = X_test_sub

    franklin = args.franklin
    trainsource = args.trainsource

    # Shuffle data if desired
    if args.shuffle:
        print(f"X length: {len(X_train)}, y length: {len(y_train)}")
        X_train, y_train = shuffle_data(X_train, y_train, seed=5555)

    # Subset to max number of training observations
    full_data_used = True
    if args.n < len(X_train):
        full_data_used = False

    if args.skiplargen:
        n = args.n
    else:
        n = min(args.n, len(X_train)) # if user entered more than length of training data, then all training observations used

    if args.n <= len(X_train):
        n = args.n
        X_train = X_train[:n, :]
        y_train = y_train[:n]

        # Get model type and hyperparameters
        modeltype = args.modeltype

        n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features = args.n_estimators, args.max_depth, args.min_samples_split, args.min_samples_leaf, args.max_features
        alpha = args.alpha
        ocr_threshold = args.ocrthreshold

        num_cv_splits = args.cvsplits

        # Print some important outputs as sanity check
        print(f"\nShape of X_train = {X_train.shape}, shape of y_train = {y_train.shape}\n")
        print(f"\nShape of X_test = {X_test.shape}, shape of y_test = {y_test.shape}\n")

        comments = args.comments
        modeltype = args.modeltype
        num_cv_splits = args.cvsplits
        n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features = args.n_estimators, args.max_depth, args.min_samples_split, args.min_samples_leaf, args.max_features
        alpha = args.alpha
        n = args.n
        seed = args.seed
        keep = args.keep
        ocr_threshold = args.ocrthreshold

        if args.testsource == "segmentation_errors":
            X_test = X_test_segmentation_error
            y_test = y_test_segmentation_error
            print(f"Using segmentation error labels as test set X_test = {X_test.shape}, shape of y_test = {y_test.shape}\n")

        # Run the experiment
        run_experiment(modeltype, n, trainsource, full_data_used, keep, X_train, X_test, y_train, y_test,
                    franklin, X_franklin_1920, y_franklin_1920, X_franklin_1931, y_franklin_1931,
                    colnames, num_cv_splits, seed,
                    n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,
                    alpha, ocr_threshold, comments)
