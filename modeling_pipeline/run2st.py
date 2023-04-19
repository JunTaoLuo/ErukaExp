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

from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

from c2st import c2st


def run_experiment(sample0_name, X_sample0, sample1_name, X_sample1, comments=''):
    '''
    desc: stitches the whole training-prediction pipeline together as an 'experiment'

    args:
        - sample1_name: name or label of the first sample for a two sample test
        - sample2_name: name or label of the second sample for a two sample test
    '''

    print(f"Sample 0: {X_sample0.shape}")
    print(f"Sample 1: {X_sample1.shape}")

    if X_sample0.shape[1] != X_sample1.shape[1]:
        print(f"Features for the two samples to do not match: sample 0 has {X_sample0.shape[1]} features, sample 1 has {X_sample1.shape[1]} features")

    # TODO: think about some if condition here for logging only the relevant hyperparams in the config
    # and setting the rest to none. Right now, all hyperparams will log for all model types (using the default values)
    # even if it doesn't make sense. Will require the user, when analysing results, to focus on the relevant columns
    # for each model class.

    wandb.init(project='eruka-housing', entity='gormleylab',
               name=f'classifier_two_sample_test',
               config={'sample0_name': sample0_name,
                       'sample1_name': sample1_name,
                       'comments':comments})

    # Note: here, I pull the hyperparameters from the wandb init config (rather than directly using the command line variables)
    # This is to allow for wandb sweeps to be configured for hyperparamter search (where we loop across different config vals)

    y_sample0 = np.zeros((len(X_sample0), 1))
    y_sample1 = np.ones((len(X_sample1), 1))

    X_combined = np.concatenate((X_sample0, X_sample1), axis=0)
    y_combined = np.concatenate((y_sample0, y_sample1), axis=0)

    print(f"Combined X: {X_combined.shape}")
    print(f"Combined Y: {y_combined.shape}")

    emp_loss, random_losses, pvalue = c2st(X_combined, y_combined)

    # Metrics to log

    wandb.log({'emp_loss': emp_loss,
            'pvalue': pvalue,
            })

    wandb.finish()


if __name__ == '__main__':

    # Arguments to be entered through command line: see 'help' for description

    X_hand_sample = np.genfromtxt('matrices/hand_sample.txt')
    X_download_errors_sample = np.genfromtxt('matrices/download_errors_sample.txt')

    # Run the experiment
    run_experiment(sample0_name="hand_labeled", X_sample0=X_hand_sample, sample1_name="download_errors", X_sample1=X_download_errors_sample)
