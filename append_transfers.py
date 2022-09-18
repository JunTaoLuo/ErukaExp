'''
File: append_transfers.py
Author: Mihir Bhaskar
Purpose: Append the historical transfer files from 1998-2022
'''
import csv
import pandas as pd
import numpy as np
import argparse

# Specify arguments to be fed when running file
parser = argparse.ArgumentParser()
parser.add_argument('input_path', type=str,
                    help='path to folder with raw .csv property transfer files')
parser.add_argument('output_path', type=str,
                    help='file path (including filename) of final appended .csv')


def append_1998_2006(input_path):
    '''
    Input (str): file path where the raw .csv files are stored
    Output (df): pandas dataframe with all the 1998-2006 data appended 
    '''
    
    df = pd.read_csv('{}\\transfer_files_ytd_1998.csv'.format(input_path), 
                     header=None) 
    
    for i in range(1999, 2007):
        tmp = pd.read_csv('{}\\transfer_files_ytd_{}.csv'.format(input_path, i),
                          header=None)
        df = pd.concat([df, tmp])
        
    # Naming the columns in the data frame
    cols = ['book', 'plat', 'parcel', 'multi_owner', 'tax_district', 'owner_name_1',
            'owner_name_2', 'land_value', 'building_value', 'property_class', 'house_no',
            'street_name', 'street_suffix', 'month_of_sale', 'day_of_sale', 
            'year_of_sale', 'num_parcels_sold', 'sale_price', 'valid_sale', 'conveyance_no',
            'deed_type']
    
    df.columns = cols
    
    # Adding columns that are present in the 2007 schema onwards, for consistency
    df.insert(loc=13, column='zip_code', value=['' for i in range(df.shape[0])])
    df['appraisal_area'], df['prior_owner'], df['property_no'] = [np.nan, np.nan, np.nan]
    
    return df

        

def append_2007_2022(input_path):
    '''
    Input (str): file path where the raw .csv files are stored
    Output (df): pandas dataframe with all the 2007-2022 data appended 
    '''
    
    df = pd.read_csv('{}\\transfer_files_ytd_2007.csv'.format(input_path), 
             header=None) 

    for i in range(2007, 2023):
        tmp = pd.read_csv('{}\\transfer_files_ytd_{}.csv'.format(input_path, i),
                          header=None)
        df = pd.concat([df, tmp])    
        
    cols = ['book', 'plat', 'parcel', 'multi_owner', 'tax_district', 'owner_name_1',
        'owner_name_2', 'land_value', 'building_value', 'property_class', 'house_no',
        'street_name', 'street_suffix', 'zip_code', 'month_of_sale', 'day_of_sale', 
        'year_of_sale', 'num_parcels_sold', 'sale_price', 'valid_sale', 'conveyance_no',
        'deed_type', 'appraisal_area', 'prior_owner', 'property_no']
    
    df.columns=cols
        
    return df

#TODO: Cleaning functions
    
if __name__ == '__main__':
    
    # Get arguments entered 
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    
    until_2006 = append_1998_2006(input_path)
    from_2007 = append_2007_2022(input_path)
    
    # Concatenate the data frames and export the file to desired location
    final_df = pd.concat([until_2006, from_2007])
    final_df.to_csv(output_path, index=False)
        
    



