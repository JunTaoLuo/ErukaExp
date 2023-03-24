'''
fill_db_franklin
Code to source the raw data and populate the database with structured data from Franklin County (Columbus)
'''

import pandas as pd
import os
import requests
from sqlalchemy import create_engine
import ohio.ext.pandas # library with better pandas -> postgreSQL writing

def populate_db_from_url(url, table_name, db_engine):
    response = requests.get(url)
    df = pd.read_excel(response.content)
    print(f"{table_name} dataframe has {len(df)} rows")
    df.to_sql(table_name, db_engine, schema = 'raw_franklin', if_exists="replace", index=False)
    print(f"Finished writing {table_name} to the database.")

if __name__ == '__main__':

    # Get SQL connection from environment
    db_str = os.getenv('ERUKA_DB')
    db = create_engine(db_str)
    
    # Create schema
    db.execute('''drop schema if exists raw_franklin cascade;
                  create schema raw_franklin;''')
        
    # Write most recent appraisal data
    url = 'https://apps.franklincountyauditor.com/Outside_User_Files/2023/2023-03-15%20Appraisal/'
    
    for table in ['Dwelling', 'Improvement', 'Parcel','Land']:
        file_str = url + table + '.xlsx'
        populate_db_from_url(file_str, table, db)
        
    # Write most recent tax info
    url = 'https://apps.franklincountyauditor.com/Outside_User_Files/2023/2023-03-15%20Tax%20Accounting/'
    file_str = url + 'Parcel.xlsx'
    populate_db_from_url(file_str, 'Parcel_Tax', db)