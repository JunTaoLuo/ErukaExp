import pandas as pd
from sqlalchemy import create_engine, text
import yaml

from reshape_building_data import reshape_building_data

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

# Read in buildinginfo
bi = pd.read_sql('SELECT * from processed.building_info', engine)

# Make sure I drop cases where storyht in buildinginfo sets off a cleaning flag (either = 0, or sqft_flr2 > 0 and storyht <= 1)
# Drop cases wher mt.prop_class_code != hs.use_code (in historic sales)