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

def read_labels(engine, keep, ocr_path, ocr_threshold):
    '''
    args:
        - engine: sql engine object, connecting to the database
        - keep: either 'all' (all handlabeled points) or
                'simple' (remove nonstandard handlabeled cases where year is entered or card is handwritten)
        - ocr_path: filepath to the OCR predicted labels (remove when we transition this to database)
    output:
        - pandas dataframe of the hand-labled observations
        - pandas dataframe of ocr-labeled observations
        - pandas dataframes of test hand-labeled observations from Franklin cCounty, for 1920 and 1931
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

    franklin_labels_1920 = pd.read_sql('''select parcelid, building_value
                                          from franklin.building_values
                                          where year = 20 and building_value is not null''', engine)

    franklin_labels_1931 = pd.read_sql('''select parcelid, building_value
                                          from franklin.building_values_1930
                                          where building_value is not null''', engine)

    print(f"Read labels: there are {len(franklin_labels_1920)} Franklin test observations from 1920")
    print(f"Read labels: there are {len(franklin_labels_1931)} Franklin test observations from 1931")

    if keep == 'all':
        sql = '''SELECT parcelid, building_value,
                        COALESCE(year, 1933) as "appraisal_target_year",
                        case when year is null then 0 else 1 end as "appraisal_target_year_flag",
                        case when handwritten is True then 1 else 0 end as "appraisal_handwritten_flag"
                 FROM samples.segmentation_error_values WHERE building_value is not null
              '''

    if keep == 'simple':
        sql = '''SELECT parcelid, building_value
                 FROM samples.segmentation_error_values
                 WHERE year is null and handwritten is null and building_value is not null'''

    segmentation_error_labels = pd.read_sql(sql, engine)
    print(f"Read labels: there are {len(segmentation_error_labels)} hand-labeled observations for segmentation errors")

    ocr_labels = pd.read_csv(ocr_path)
    print(f"Read labels: there are {len(ocr_labels)} OCR-labeled observations")

    ocr_labels['prediction'] = pd.to_numeric(ocr_labels['prediction'], errors='coerce') # conver prediction to numeric

    # Filter out OCR predictions based on conditions:

    ## 1. Remove parcelids for which we already have hand-labeled data
    ocr_labels = ocr_labels[~ocr_labels['parcelid'].isin(hand_labels['parcelid'])]

    ## 2. Keep only cases for which we have nonmissing predictions
    ocr_labels = ocr_labels[~ocr_labels['prediction'].isna()]

    ## 3. Keep only cases where our prediction confidence is above the set threshold (see OCR modeling work for how we arrived at this)
    ocr_labels = ocr_labels[ocr_labels['score'] >= ocr_threshold]

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
    segmentation_error_labels['is_ocr'] = 0

    franklin_labels_1920['is_ocr'] = 0
    franklin_labels_1931['is_ocr'] = 0

    return hand_labels, ocr_labels, franklin_labels_1920, franklin_labels_1931, segmentation_error_labels

def read_features(engine):
    '''
    args:
        - engine: engine object, connecting to the database
    output:
        - pandas dataframe of: all franklin features, and a subset used for generalization to Franklin
    todo:
        - make feature choices into an argument, that can be populated from argparse
        to try and experiment with different feature choices
    other notes:
        - see Eruka Features sheet on GDrive for discussion on each feature and choices
    '''

    # SQL code to select all possible relevant variables from the database
    sql = '''
            SELECT parcelid,
                    bi.attic_sqft, bi.attic_cat, bi.bsmt_sqft, bi.live_sqft, bi.sqft_flr1,
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

    # Subset features to be used for franklin generalizations
    feats_sub = pd.read_sql(sql, engine)

    common_feats = ['parcelid', 'attic_cat', 'live_sqft', 'sqft_flr1', 'story_ht', 'year_built',
                   'prop_class_code', 'number_of_parcels', 'grade_numeric', 'exterior_wall_type', 'basement_grouped',
                   'heating', 'air_conditioning', 'total_rooms', 'full_bath', 'half_bath', 'fireplaces', 'garage_capacity']

    feats_sub = feats_sub[common_feats]

    feats_sub = pd.get_dummies(feats_sub, columns=['attic_cat', 'prop_class_code',
                                         'exterior_wall_type', 'heating', 'air_conditioning', 'basement_grouped'], dummy_na = True)

    feats_franklin = pd.read_sql('SELECT * FROM processed.franklin_features', engine)

    feats_franklin = feats_franklin[common_feats]

    feats_franklin = pd.get_dummies(feats_franklin, columns=['attic_cat', 'prop_class_code',
                                         'exterior_wall_type', 'heating', 'air_conditioning', 'basement_grouped'], dummy_na = True)

    # Because creating dummy variables might result in different numbers of columns, aligning the X data frames

    # Keep only columns that exist across both data frames
    common_cols = feats_franklin.columns.intersection(feats_sub.columns)
    feats_franklin = feats_franklin.loc[:, common_cols]
    feats_sub = feats_sub.loc[:, common_cols]

    return feats, feats_sub, feats_franklin

def read_data(engine, keep, ocr_path, ocr_threshold):
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

    hand_labels, ocr_labels, franklin_labels_1920, franklin_labels_1931, segmentation_error_labels = read_labels(engine, keep, ocr_path, ocr_threshold)
    features, feats_sub, feats_franklin = read_features(engine)

    hand_merged = pd.merge(hand_labels, features, on='parcelid')
    segmentation_error_merged = pd.merge(segmentation_error_labels, features, on='parcelid')

    hand_merged_sub = pd.merge(hand_labels, feats_sub, on='parcelid')
    franklin_1920_merged = pd.merge(franklin_labels_1920, feats_franklin, on='parcelid')
    franklin_1931_merged = pd.merge(franklin_labels_1931, feats_franklin, on='parcelid')


    if len(hand_labels) != len(hand_merged):
        warnings.warn(f"There are {len(hand_labels) - len(hand_merged)} handlabeled parcels getting dropped when merging to feature data.")

    if hand_merged['year_built'].isna().sum() > 0:
        warnings.warn(f"There is missing feature data for handlabeled parcels where there shouldn't be.")

    print(f"Merged hand labels and features. The data shape is {hand_merged.shape}")

    if len(segmentation_error_labels) != len(segmentation_error_merged):
        warnings.warn(f"There are {len(segmentation_error_labels) - len(segmentation_error_merged)} segmentation_error_labeled parcels getting dropped when merging to feature data.")

    if segmentation_error_merged['year_built'].isna().sum() > 0:
        warnings.warn(f"There is missing feature data for handlabeled parcels where there shouldn't be.")

    print(f"Merged segmentation error labels and features. The data shape is {segmentation_error_merged.shape}")

    ocr_merged = pd.merge(ocr_labels, features, on='parcelid')
    ocr_merged_sub = pd.merge(ocr_labels, feats_sub, on='parcelid')

    if len(ocr_labels) != len(ocr_merged):
        warnings.warn(f"There are {len(ocr_labels) - len(ocr_merged)} ocr parcels getting dropped when merging to feature data.")

    if ocr_merged['year_built'].isna().sum() > 0:
        warnings.warn(f"There is missing feature data for ocr parcels where there shouldn't be.")

    print(f"Merged OCR labels and features. The data shape is {ocr_merged.shape}")

    return hand_merged, ocr_merged, hand_merged_sub, ocr_merged_sub, franklin_1920_merged, franklin_1931_merged, segmentation_error_merged

def extract_xy(df):
    y = df['building_value']
    X = df.drop(columns=['parcelid', 'building_value'])
    return X, y

def split_data(df, test_size = 0.2, random_state = 4):
    '''
    args:
        - df: pandas dataframe of the labeled data merged with features
        - test_size: proportion of observations to split into test set
        - random_state: random seet for splitting data
    outputs:
        - X_train, X_test, y_train, y_test: pandas dfs of training and test features and labels
    '''

    X, y = extract_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

    print(f'''Data was split into into X, y, train and test. The shapes are:
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

    # Impute building height by modal value (2 floors)
    X_df['story_ht'] = X_df['story_ht'].fillna(2)

    # Impute total rooms by modal value (6 floors)
    X_df['total_rooms'] = X_df['total_rooms'].fillna(6)

    # Impute with logical value if missing (0)
    X_df['full_bath'] = X_df['full_bath'].fillna(0)
    X_df['half_bath'] = X_df['half_bath'].fillna(0)
    X_df['fireplaces'] = X_df['fireplaces'].fillna(0)
    X_df['garage_capacity'] = X_df['garage_capacity'].fillna(0)

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

def gen_matrix(df, matrix_path, matrix_name):
    if not isinstance(df, np.ndarray):
        df = df.to_numpy()

    np.savetxt(f"{matrix_path}/{matrix_name}.txt", df)

    print(f"{matrix_name} saved to file")

    return df

def main(engine, ocr_threshold, keep='simple', ocr_path='oc-carb-fine-tuning-10k_results.csv', test_size=0.2, random_state=4, matrix_path='matrices'):
    '''
    Main function that stitches everything in this file together, to generate the
    train and test matrices written to file.

    Args are documented in their individual functions defined above.
    '''
    hand_df, ocr_df, hand_df_sub, ocr_df_sub, franklin_1920_df, franklin_1931_df, segmentation_error_df = read_data(engine, keep, ocr_path, ocr_threshold=ocr_threshold)

    if keep=='all':

        # If all observations kept, only make train-test split based on cases that are no year + not handwritten, so test
        # set only contains observations like that (our ground truth)
        hand_df_simple = hand_df[(hand_df['appraisal_target_year'] == 1933) & (hand_df['appraisal_handwritten_flag'] == 0)]

        X_train_hand, X_test, y_train_hand, y_test = split_data(hand_df_simple, test_size, random_state)

        hand_df_simple_sub = hand_df_sub[(hand_df_sub['appraisal_target_year'] == 1933) & (hand_df_sub['appraisal_handwritten_flag'] == 0)]

        X_train_hand_sub, X_test_sub, y_train_hand_sub, y_test_sub = split_data(hand_df_simple_sub, test_size, random_state)

        # Concat the rest of the data (i.e., cases with year or handwritten) to training only
        hand_df_yearhand = hand_df[~hand_df['parcelid'].isin(hand_df_simple['parcelid'])]
        y_train_yearhand = hand_df_yearhand['building_value']
        X_train_yearhand = hand_df_yearhand.drop(columns=['parcelid', 'building_value'])

        X_train_hand = pd.concat([X_train_hand, X_train_yearhand])
        y_train_hand = pd.concat([y_train_hand, y_train_yearhand])

        hand_df_yearhand_sub = hand_df_sub[~hand_df_sub['parcelid'].isin(hand_df_simple_sub['parcelid'])]
        X_train_yearhand_sub = hand_df_yearhand_sub.drop(columns=['parcelid', 'building_value'])

        X_train_hand_sub = pd.concat([X_train_hand_sub, X_train_yearhand_sub])

        assert y_train_hand_sub.equals(y_train_hand)
        assert y_test_sub.equals(y_test)

        segmentation_error_df_simple = segmentation_error_df[(segmentation_error_df['appraisal_target_year'] == 1933) & (segmentation_error_df['appraisal_handwritten_flag'] == 0)]
        X_test_segmentation_error, y_test_segmentation_error = extract_xy(segmentation_error_df_simple)

    else:
        # Else, since all handlabeled data has no year/is not handwritten, can use train-test split directly
        X_train_hand, X_test, y_train_hand, y_test = split_data(hand_df, test_size, random_state)

        X_train_hand_sub, X_test_sub, y_train_hand_sub, y_test_sub = split_data(hand_df_sub, test_size, random_state)

        X_test_segmentation_error, y_test_segmentation_error = extract_xy(segmentation_error_df)

        assert y_train_hand_sub.equals(y_train_hand)
        assert y_test_sub.equals(y_test)

    X_train_hand, colnames = process_features(X_train_hand)
    X_test, colnames = process_features(X_test)

    X_train_hand_sub, colnames_sub = process_features(X_train_hand_sub)
    X_test_sub, colnames_sub = process_features(X_test_sub)

    X_train_ocr = ocr_df.drop(columns=['parcelid', 'building_value'])
    y_train_ocr = ocr_df['building_value']
    X_train_ocr, colnames = process_features(X_train_ocr)

    X_train_ocr_sub = ocr_df_sub.drop(columns=['parcelid', 'building_value'])
    X_train_ocr_sub, colnames_sub = process_features(X_train_ocr_sub)

    X_franklin_1920 = franklin_1920_df.drop(columns=['parcelid', 'building_value'])
    X_franklin_1920, colnames_sub = process_features(X_franklin_1920)
    X_franklin_1931 = franklin_1931_df.drop(columns=['parcelid', 'building_value'])
    X_franklin_1931, colnames_sub = process_features(X_franklin_1931)

    y_franklin_1920 = franklin_1920_df['building_value']
    y_franklin_1931 = franklin_1931_df['building_value']

    X_test_segmentation_error, _ = process_features(X_test_segmentation_error)

    mapping = {
        'X_train_hand': X_train_hand,
        'X_train_ocr': X_train_ocr,
        'X_test': X_test,
        'X_train_hand_sub': X_train_hand_sub,
        'X_train_ocr_sub': X_train_ocr_sub,
        'X_test_sub': X_test_sub,
        'y_train_hand': y_train_hand,
        'y_train_ocr': y_train_ocr,
        'y_test': y_test,
        'X_franklin_1920': X_franklin_1920,
        'X_franklin_1931': X_franklin_1931,
        'y_franklin_1920': y_franklin_1920,
        'y_franklin_1931': y_franklin_1931,
        'X_test_segmentation_error': X_test_segmentation_error,
        'y_test_segmentation_error': y_test_segmentation_error,
    }

    for i in mapping:
        gen_matrix(mapping[i], matrix_path, i)

    # Export colnames
    with open(f"{matrix_path}/colnames.txt", 'w') as file:
        file.write('\n'.join(colnames))

    with open(f"{matrix_path}/colnames_sub.txt", 'w') as file:
        file.write('\n'.join(colnames_sub))

    print("Column names saved to file")

    return X_train_hand, X_train_ocr, X_test, X_train_hand_sub, X_train_ocr_sub, X_test_sub, y_train_hand, y_train_ocr, y_test, X_franklin_1920, X_franklin_1931, y_franklin_1920, y_franklin_1931, X_test_segmentation_error, y_test_segmentation_error, colnames, colnames_sub



# def gen_matrices(X_train_hand, X_train_ocr, X_test, y_train_hand, y_train_ocr, y_test, colnames, matrix_path='matrices'):
#     '''
#     args:
#         - train and test pandas dataframes or numpy arrays ready for modeling. Training data options for for hand-labeled only, and ocr-only.
#           test data only comes from hand-labeled data.
#         - colnames: column headers as a list, to write to a txt file (for postmodeling)
#         - matrix_path: path to write the matrices and column headers to
#     outputs:
#         - train and test matrices as numpy arrays, written as txt files and returned. Both options of hand-labeled only and ocr-only are exported
#         - colnames list written as a txt file to the specified path
#     purpose:
#         - arrays are faster than pandas dfs for processing
#     '''
#     for df in [X_train_hand, X_train_ocr, y_train_hand, y_train_ocr, X_test, y_test]:
#         if not isinstance(df, np.ndarray):
#             df = df.to_numpy()

#     np.savetxt(f"{matrix_path}/X_train_hand.txt", X_train_hand)
#     np.savetxt(f"{matrix_path}/X_train_ocr.txt", X_train_ocr)
#     np.savetxt(f"{matrix_path}/X_test.txt", X_test)
#     np.savetxt(f"{matrix_path}/y_train_hand.txt", y_train_hand)
#     np.savetxt(f"{matrix_path}/y_train_ocr.txt", y_train_ocr)
#     np.savetxt(f"{matrix_path}/y_test.txt", y_test)

#     print("Matrices saved to file")

#     # Export colnames
#     with open(f"{matrix_path}/colnames.txt", 'w') as file:
#         file.write('\n'.join(colnames))

#     print("Column names saved to file")

#     return X_train_hand, X_train_ocr, X_test, y_train_hand, y_train_ocr, y_test





