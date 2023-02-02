import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def read_labels(engine, keep):
    '''
    args:
        - sql engine object
        - keep: either 'all' (all labeled points) or 
                'simple' (remove nonstandard cases where year is entered or card is handwritten)
    '''

    if keep == 'all':
        sql = 'SELECT * FROM samples.building_values WHERE building_value is not null'

    if keep == 'simple':
        sql = '''SELECT parcelid, building_value
                 FROM samples.building_values 
                 WHERE year is null and handwritten is null and building_value is not null'''

    return pd.read_sql(sql, engine)

def read_features(engine):
    sql = '''
            SELECT parcelid,
                    bi.attic_sqft, bi.bsmt_sqft, bi.live_sqft, bi.sqft_flr1, 
                    bi.sqft_flr2, bi.sqft_flrh, bi.story_ht, bi.year_built, bi.prop_class_code, bi.class_description,

                    hs.number_of_parcels, hs.date_of_sale, hs.appraisal_area, hs.total_sales_records, hs.sale_price,
                    hs.style, hs.grade, hs.exterior_wall_type, hs.basement, hs.heating, hs.air_conditioning, hs.total_rooms,
                    hs.full_bath, hs.half_bath, hs.fireplaces, hs.garage_type, hs.finished_sq_ft, hs.total_finish_area,
                    hs.first_floor_area, hs.half_floor_area, hs.finished_basement

            FROM processed.building_info bi JOIN processed.historic_sales hs USING(parcelid)
         '''
    
    return pd.read_sql(sql, engine)
                

def read_data(engine, keep):
    labels = read_labels(engine, keep)
    features = read_features(engine)

    merged = pd.merge(labels, features, on='parcelid')

    if len(labels) != len(merged):
        warnings.warn(f"There are {len(labels) - len(merged)} labeled rows getting dropped when merging to feature data.") 
    
    if merged['prop_class_code'].isna().sum() > 0:
        warnings.warn(f"There is missing feature data for labels where there shouldn't be.")

    return merged

def split_data(df, test_size = 0.2, random_state = 4):
    y = df['building_value']
    X = df.drop(columns=['parcelid', 'building_value', 'date_of_sale', 'class_description']) # can consider doing something with date_of_sale later

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

    return X_train, X_test, y_train, y_test

def process_features(X_df):
        
    # Todo: convert categorical variables to one-hot, after doing some clubbing
    # style, grade, exterior wall type, basement, heating, aircon, garage type

    # keep only numeric features so we don't run into problems with different classes (e.g. linear regression),
    # but can relax this later
    X_df = X_df.select_dtypes(['number'])

    # use a simple imputation to address missing values (do something more sophisticated later)
    imp = SimpleImputer(strategy='mean')
    X_df = imp.fit_transform(X_df)

    # use a standard scaler so we don't run into issues with regularization
    scaler = StandardScaler().fit(X_df)
    X_df = scaler.transform(X_df)

    return X_df
    
