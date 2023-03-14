'''
prep_data.py
Author: Mihir Bhaskar
Purpose: pipeline for creating the train-test matrices for modeling

Notes: for now, I log using print statements to command line because the pipeline
is very simple and local. If it gets more compliated, will start persisting and printing
to a log file.
'''
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def read_labels(engine, keep, ocr_path):
    '''
    args:
        - engine: sql engine object, connecting to the database
        - keep: either 'all' (all handlabeled points) or 
                'simple' (remove nonstandard handlabeled cases where year is entered or card is handwritten)
        - ocr_path: filepath to the OCR predicted labels (remove when we transition this to database)
    output:
        - pandas dataframe of the hand-labled observations
        - pandas dataframe of ocr-labeled observations
    '''

    if keep == 'all':
        sql = '''SELECT parcelid, building_value,
                        COALESCE(year, 1933) as "appraisal_target_year",
                        case when year is null then 0 else 1 end as "appraisal_target_year_flag",
                        case when handwritten is True then 1 else 0 end as "appraisal_handwritten_flag"
                 FROM samples.building_values WHERE building_value is not null
              '''
              
    if keep == 'simple':
        sql = '''SELECT parcelid, building_value
                 FROM samples.building_values 
                 WHERE year is null and handwritten is null and building_value is not null'''
                 
    hand_labels = pd.read_sql(sql, engine)
    print(f"Read labels: there are {len(hand_labels)} hand-labeled observations")
                 
    ocr_labels = pd.read_csv(ocr_path)
    print(f"Read labels: there are {len(ocr_labels)} OCR-labeled observations") 
    
    ocr_labels['prediction'] = pd.to_numeric(ocr_labels['prediction'], errors='coerce') # conver prediction to numeric
    
    # Filter out OCR predictions based on conditions:
    
    ## 1. Remove parcelids for which we already have hand-labeled data
    ocr_labels = ocr_labels[~ocr_labels['parcelid'].isin(hand_labels['parcelid'])]
    
    ## 2. Keep only cases for which we have nonmissing predictions
    ocr_labels = ocr_labels[~ocr_labels['prediction'].isna()]
    
    ## 3. Keep only cases where our prediction confidence is above the set threshold (see OCR modeling work for how we arrived at this)
    ocr_labels = ocr_labels[ocr_labels['score'] >= -0.0772]
    
    ## 4. Drop very large predicted values (to reduce noise, we also don't care much about these properties even if the prediction is right)
    ocr_labels = ocr_labels[ocr_labels['prediction'] <= 50000] 
    
    print(f"There are {len(ocr_labels)} OCR-labeled observations after dropping points based on the defined logic.") 

    # Process/format OCR labels to match hand labels
    ocr_labels = ocr_labels[['parcelid', 'prediction']]
    ocr_labels.rename({"prediction": "building_value"}, axis=1, inplace=True)
    
    if keep == 'all':
        # Add columns to make OCR consistent with hand labeled data
        ocr_labels['appraisal_target_year'] = 1933
        ocr_labels['appraisal_target_year_flag'] = 0 # assuming card has no year - refine later based on OCR work
        ocr_labels['appraisal_handwritten_flag'] = 0 # assuming here that card was not handwritten - refine later based on OCR work
        
    # Adding a var that will help us keep track of whether a sample is hand-labeled or OCR. Models with just either of these data types
    # will ignore this var because it is the same for all data points, but combined models may use this to weight hand-labeled data more
    ocr_labels['is_ocr'] = 1
    hand_labels['is_ocr'] = 0
            
    return hand_labels, ocr_labels
    
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
                    hs.number_of_parcels, hs.date_of_sale, hs.appraisal_area, hs.area_description_grouped, hs.total_sales_records, hs.sale_price,
                    hs.style, hs.grade, hs.exterior_wall_type, hs.basement, hs.heating, hs.air_conditioning, hs.total_rooms,
                    hs.full_bath, hs.half_bath, hs.fireplaces, hs.garage_type, hs.finished_sq_ft, hs.total_finish_area,
                    hs.first_floor_area, hs.half_floor_area, hs.finished_basement, hs.garage_capacity, 
                    hs.grade_grouped, hs.grade_numeric, hs.basement_grouped, hs.garage_type_grouped
            FROM processed.building_info bi JOIN processed.historic_sales hs USING(parcelid)
            WHERE parcelid IN (SELECT parcelid FROM samples.labels)
         '''
         
    feats = pd.read_sql(sql, engine)
    
    # Dropping columns we don't want to use for analysis (justifications in the GDrive Eruka Features sheet)
    feats = feats.drop(['live_sqft', 'class_description', 'date_of_sale', 'total_sales_records',
                        'sale_price', 'grade_grouped', 'grade', 'basement', 'garage_type',
                        'finished_sq_ft', 'total_finish_area', 'first_floor_area', 'appraisal_area'], axis=1)
    
    # Converting to dummy variables here before merging and splitting data because we want the same # of features in train and test
    # Basically, this isn't sensitive to whether certain classes are observed in certain splits
    feats = pd.get_dummies(feats, columns=['prop_class_code', 'style',
                                         'exterior_wall_type', 'heating', 'air_conditioning', 'basement_grouped',
                                         'garage_type_grouped', 'area_description_grouped'], dummy_na = True)
    
    print(f"Finished reading features: {feats.shape[1]} columns were read in")
    
    return feats
                
def read_data(engine, keep, ocr_path):
    '''
    args:
        - engine: sql engine connecting to the database
        - keep: either 'all' (all labeled points) or 
                'simple' (remove nonstandard cases where year is entered or card is handwritten)
        - ocr_preds_path: filepath to the OCR predicted labels 
    output:
        - pandas dataframe of the hand labeled observations merged with feature data
        - pandas datarame of ocr labeled observations merged with feature data (if use_ocr is true)
    '''
    
    hand_labels, ocr_labels = read_labels(engine, keep, ocr_path)
    features = read_features(engine)
    
    hand_merged = pd.merge(hand_labels, features, on='parcelid')
    
    if len(hand_labels) != len(hand_merged):
        warnings.warn(f"There are {len(hand_labels) - len(hand_merged)} handlabeled parcels getting dropped when merging to feature data.") 
        
    if hand_merged['year_built'].isna().sum() > 0:
        warnings.warn(f"There is missing feature data for handlabeled parcels where there shouldn't be.")
            
    print(f"Merged hand labels and features. The data shape is {hand_merged.shape}")
    
    ocr_merged = pd.merge(ocr_labels, features, on='parcelid')
        
    if len(ocr_labels) != len(ocr_merged):
        warnings.warn(f"There are {len(ocr_labels) - len(ocr_merged)} ocr parcels getting dropped when merging to feature data.") 
    
    if ocr_merged['year_built'].isna().sum() > 0:
        warnings.warn(f"There is missing feature data for ocr parcels where there shouldn't be.")
        
    print(f"Merged OCR labels and features. The data shape is {ocr_merged.shape}")
        
    return hand_merged, ocr_merged
            
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

    print(f'''Handlabeled data was split into into X, y, train and test. The shapes are:
              \nX_train: {X_train.shape}
              \nX_test: {X_test.shape}
              \ny_train: {y_train.shape}
              \ny_test: {y_test.shape}''')

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
    
    print("Features were imputed!")
    
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
    
    print("Features were processed!")

    return X_df, colnames

def gen_matrices(X_train_hand, X_train_ocr, X_test, y_train_hand, y_train_ocr, y_test, colnames, matrix_path='matrices'):
    '''
    args:
        - train and test pandas dataframes or numpy arrays ready for modeling. Training data options for for hand-labeled only, and ocr-only.
          test data only comes from hand-labeled data.
        - colnames: column headers as a list, to write to a txt file (for postmodeling)
        - matrix_path: path to write the matrices and column headers to
    outputs:
        - train and test matrices as numpy arrays, written as txt files and returned. Both options of hand-labeled only and ocr-only are exported
        - colnames list written as a txt file to the specified path
    purpose:
        - arrays are faster than pandas dfs for processing
    '''
    for df in [X_train_hand, X_train_ocr, y_train_hand, y_train_ocr, X_test, y_test]:
        if not isinstance(df, np.ndarray):
            df = df.to_numpy()
            
    np.savetxt(f"{matrix_path}/X_train_hand.txt", X_train_hand)
    np.savetxt(f"{matrix_path}/X_train_ocr.txt", X_train_ocr)
    np.savetxt(f"{matrix_path}/X_test.txt", X_test)
    np.savetxt(f"{matrix_path}/y_train_hand.txt", y_train_hand)
    np.savetxt(f"{matrix_path}/y_train_ocr.txt", y_train_ocr)
    np.savetxt(f"{matrix_path}/y_test.txt", y_test)
    
    print("Matrices saved to file")
    
    # Export colnames
    with open(f"{matrix_path}/colnames.txt", 'w') as file:
        file.write('\n'.join(colnames))
    
    print("Column names saved to file")
    
    return X_train_hand, X_train_ocr, X_test, y_train_hand, y_train_ocr, y_test
    
def main(engine, keep='simple', ocr_path='oc-carb-fine-tuning-10k_results.csv', test_size=0.2, random_state=4, matrix_path='matrices'):
    '''
    Main function that stitches everything in this file together, to generate the
    train and test matrices written to file. 
    
    Args are documented in their individual functions defined above.
    '''
    hand_df, ocr_df = read_data(engine, keep, ocr_path)
    
    # Splitting hand-labeled data into train and test, processing features
    X_train_hand, X_test, y_train_hand, y_test = split_data(hand_df, test_size, random_state)
    X_train_hand = impute_features(X_train_hand)
    X_train_hand, colnames = process_features(X_train_hand)
    X_test = impute_features(X_test)
    X_test, colnames = process_features(X_test)
    
    X_train_ocr = ocr_df.drop(columns=['parcelid', 'building_value'])
    y_train_ocr = ocr_df['building_value']
    # X_train_ocr = impute_features(X_train_ocr)
    X_train_ocr, colnames = process_features(X_train_ocr)
    
    X_train_hand, X_train_ocr, X_test, y_train_hand, y_train_ocr, y_test = gen_matrices(X_train_hand, X_train_ocr, X_test, y_train_hand, y_train_ocr, y_test, colnames, matrix_path)
    
    return X_train_hand, X_train_ocr, X_test, y_train_hand, y_train_ocr, y_test, colnames


    
    
    
