## Modeling pipeline for Eruka: Historical Building Value Estimation (work-in-progress)

### Overview
This pipeline consists of the following:
1. `prep_data.py`: Functions to read and process label and features data from the postgreSQL database, converting them to train-test matrices
2. `run.py`: The main pipeline runner, which takes several command line arguments to decide how to process the data, which model to run, and which metrics/graphs to create/log
3. Logged graphs and outputs/results are written to this project's page at [WeightsandBiases](https://wandb.ai/home)

The `sweep_configs` contains .yaml config files that are used for grid searches across hyperparamters, or any other varying specifications that need to be run in a loop. They use WeightsandBiases' *sweep* functionality.

The `test_modeling.ipynb` notebook is a playground to test out different functions, graphs, and more before converting them to functions in the pipeline.


### How to run
1. Ensure requirements are installed from `requirements.txt` at the root of the repo
2. Ensure that the database credentials/URI is stored as an environment variable called `ERUKA_DB`
3. Log in to wandb using the command line
4. Execute `run.py` from the command line, with relevant arguments telling it what to run (see `run.py` file for more details)
5. Go to the wandb project page to check out the results

For sweeps/grid searches:
1. Specify the parameters to vary in a .yaml config file (see current examples)
2. Initialize and run a wandb sweep with that .yaml config file (see wandb documentation on this)