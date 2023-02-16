'''
prep_data.py
Author: Mihir Bhaskar
Purpose: pipeline for creating the train-test matrices for modeling
'''
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#TODO: add logging statements

def read_labels(engine, keep):
    '''
    args:
        - engine: sql engine object, connecting to the database
        - keep: either 'all' (all labeled points) or 
                'simple' (remove nonstandard cases where year is entered or card is handwritten)
    output:
        - pandas dataframe of the labeled observations
    '''

    if keep == 'all':
        sql = 'SELECT * FROM samples.building_values WHERE building_value is not null'

    if keep == 'simple':
        sql = '''SELECT parcelid, building_value
                 FROM samples.building_values 
                 WHERE year is null and handwritten is null and building_value is not null'''

    return pd.read_sql(sql, engine)

def read_features(engine):
    '''
    args:
        - engine: engine object, connecting to the database
    output:
        - pandas dataframe of all the features
    todo:
        - make feature choices into an argument, that can be populated from argparse
        to try and experiment with different feature choices
    other notes: 
        - see Eruka Features sheet on GDrive for discussion on each feature and choices
    '''
    
    # SQL code to select all possible relevant variables from the database
    sql = '''
            SELECT parcelid,
                    bi.attic_sqft, bi.bsmt_sqft, bi.live_sqft, bi.sqft_flr1, 
                    bi.sqft_flr2, bi.sqft_flrh, bi.story_ht, bi.year_built, bi.prop_class_code, bi.class_description,
                    bi.live_sqft_prop,

                    hs.number_of_parcels, hs.date_of_sale, hs.appraisal_area, hs.total_sales_records, hs.sale_price,
                    hs.style, hs.grade, hs.exterior_wall_type, hs.basement, hs.heating, hs.air_conditioning, hs.total_rooms,
                    hs.full_bath, hs.half_bath, hs.fireplaces, hs.garage_type, hs.finished_sq_ft, hs.total_finish_area,
                    hs.first_floor_area, hs.half_floor_area, hs.finished_basement, hs.garage_capacity, 
                    hs.grade_grouped, hs.grade_numeric, hs.basement_grouped, hs.garage_type_grouped

            FROM processed.building_info bi JOIN processed.historic_sales hs USING(parcelid)
         '''
         
    feats = pd.read_sql(sql, engine)
    
    # Dropping columns we don't want to use for analysis (justifications in the GDrive Eruka Features sheet)
    feats = feats.drop(['live_sqft', 'class_description', 'date_of_sale', 'total_sales_records',
                        'sale_price', 'grade_grouped', 'grade', 'basement', 'garage_type',
                        'finished_sq_ft', 'total_finish_area', 'first_floor_area'], axis=1)
    
    return feats
                
def read_data(engine, keep):
    '''
    args:
        - engine: sql engine connecting to the database
        - keep: either 'all' (all labeled points) or 
                'simple' (remove nonstandard cases where year is entered or card is handwritten)
    output:
        - pandas dataframe of the labeled observations merged with feature data
    '''
    
    labels = read_labels(engine, keep)
    features = read_features(engine)

    merged = pd.merge(labels, features, on='parcelid')

    if len(labels) != len(merged):
        warnings.warn(f"There are {len(labels) - len(merged)} labeled rows getting dropped when merging to feature data.") 
    
    if merged['year_built'].isna().sum() > 0:
        warnings.warn(f"There is missing feature data for labels where there shouldn't be.")
        
    # Converting to dummy variables here before splitting data because we want same # of features in train and test
    merged = pd.get_dummies(merged, columns=['prop_class_code', 'style', 'appraisal_area',
                                         'exterior_wall_type', 'heating', 'air_conditioning', 'basement_grouped',
                                         'garage_type_grouped'], dummy_na = True)
        
    return merged

def split_data(df, test_size = 0.2, random_state = 4):
    '''
    args:
        - df: pandas dataframe of the labeled data merged with features
        - test_size: proportion of observations to split into test set
        - random_state: random seet for splitting data
    outputs:
        - X_train, X_test, y_train, y_test: pandas dfs of training and test features and labels
    '''
    
    y = df['building_value']
    X = df.drop(columns=['parcelid', 'building_value'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

    return X_train, X_test, y_train, y_test

def impute_features(X_df):
    '''
    args:
        - X_df: pandas dataframe of features (could be train or test)
    outputs:
        - pandas dataframe of X_df with missing values imputed
    '''
    
    # Imputing number of parcels by modal value (1 parcel)
    X_df['number_of_parcels'] = X_df['number_of_parcels'].fillna(1) 
    
    # Note: as observed in the feature sheet, because of the cleaning and cohort selection process,
    # there aren't many features with missing values in the relevant data. We can consider things
    # like median/mode imputation
    
    return X_df
    

def process_features(X_df):
    '''
    args:
        - X_df: pandas dataframe of features (could be train or test)
    outputs:
        - numpy ndarray of X_df with feature transformations that make it ready for 
        modeling (e.g., imputation, converting categorical vars to dummies, scaling) 
        - list of all the feature names in column order (so we can add this back post-modeling)
    '''
    
    # Pull out column names
    colnames = X_df.columns.values
    
    # Imputing features as defined in function above
    X_df = impute_features(X_df)
    
    # keep only numeric features so we don't run into problems with different classes (e.g. linear regression),
    # but can relax this later
    X_df = X_df.select_dtypes(['number'])

    # use a standard scaler so we don't run into scale issues with regularization
    scaler = StandardScaler().fit(X_df)
    X_df = scaler.transform(X_df)

    return X_df, colnames

def gen_matrices(X_train, X_test, y_train, y_test, colnames, matrix_path='matrices'):
    '''
    args:
        - train and test pandas dataframes or numpy arrays ready for modeling
        - colnames: column headers as a list, to write to a txt file (for postmodeling)
        - matrix_path: path to write the matrices and column headers to
    outputs:
        - train and test matrices as numpy arrays, written as txt files and returned
        - colnames list written as a txt file to the specified path
    purpose:
        - arrays are faster than pandas dfs for processing
    '''
    # TODO: Couldn't find a good way to loop this stuff, but explore - maybe locals?
    # Put an if-condition (X_dfs are probably already numpy arrays)
   # X_train = np.to_numpy(X_train)
   # X_test = np.to_numpy(X_test)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    np.savetxt(f"{matrix_path}/X_train.txt", X_train)
    np.savetxt(f"{matrix_path}/X_test.txt", X_test)
    np.savetxt(f"{matrix_path}/y_train.txt", y_train)
    np.savetxt(f"{matrix_path}/y_test.txt", y_test)
    
    # Export colnames
    with open(f"{matrix_path}/colnames.txt", 'w') as file:
        file.write('\n'.join(colnames))
    
    return X_train, X_test, y_train, y_test
    
def main(engine, keep='simple', test_size=0.2, random_state=4, matrix_path='matrices'):
    '''
    Main function that stitches everything in this file together, to generate the
    train and test matrices written to file. 
    
    Args are documented in their individual functions defined above.
    '''
    df = read_data(engine, keep)
    X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)
    X_train, colnames = process_features(X_train)
    X_test, colnames = process_features(X_test)
    X_train, X_test, y_train, y_test = gen_matrices(X_train, X_test, y_train, y_test, colnames, matrix_path)
    
    return X_train, X_test, y_train, y_test, colnames


    
    
    
