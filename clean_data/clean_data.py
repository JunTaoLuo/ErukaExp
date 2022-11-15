'''
clean_data.py
Purpose: executes full data cleaning pipeline from raw -> processed schema ready for modeling processes
Author: Mihir Bhaskar
'''
import pandas as pd
from sqlalchemy import create_engine, text
import yaml
import ohio.ext.pandas

# Connect to database using database YAML file with connection params
with open('../database.yaml', 'r') as f:
    db_params = yaml.safe_load(f)

engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{dbname}'.format(
  host=db_params['host'],
  port=db_params['port'],
  dbname=db_params['db'],
  user=db_params['user'],
  password=db_params['pass']    
))

### Do first round basic cleaning from raw tables to create 'cleaned' schema ###
with open('r1_basic_data_cleaning.sql', 'r') as f:
    sql = f.read()

# Ensure no text problems for python
sql = text(sql)

engine.execute(sql)
engine.execute('COMMIT')

### Do second round cleaning/processing that reflects researcher decisions to create 'processed' schema ###
with open('r2_further_data_processing.sql', 'r') as f:
    sql_template = f.read()

# Parameters to fill into SQL template
class_codes = [510, 550, 520, 401, 530, 625, 404, 470] # class codes we are deciding to keep

# Format the lists before it gets inserted into the SQL template to remove the square brackets
def list_to_string(l, dtype='string'):
    if dtype=='string':
        return ','.join(["'%s'" % elm for elm in l])
    else:
        return ','.join(["%s" % elm for elm in l])

class_codes = list_to_string(class_codes)

# Insert params into SQL template and execute
sql = sql_template.format(class_codes = class_codes)

engine.execute(sql)
engine.execute('COMMIT')