import pandas as pd
from sqlalchemy import create_engine, text
import yaml

from reshape_building_data import reshape_building_data

# Connect to database using database YAML file with connection params
with open('database.yaml', 'r') as f:
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

# Keep only buildings where yearbuild is not null and yearbuild < 1930
bi_pre1930 = bi[(bi['year_built'] < 1930) & (bi['year_built'].isna() == False)]

# Drop buildings that had issues with story height
bi_pre1930 = bi_pre1930[bi_pre1930['story_ht'] != 0]

# Creating building IDs at the parcel level, that can be used in the reshape function
bi_pre1930['building_id'] = bi_pre1930.groupby('parcelid').cumcount() + 1

# Make sure I drop cases where storyht in buildinginfo sets off a cleaning flag (either = 0, or sqft_flr2 > 0 and storyht <= 1), or num_stories = 0
# Drop cases wher mt.prop_class_code != hs.use_code (in historic sales)

# Keep where yearbuild < 1930 and yearbuild is not null

# Create features of yearbuild (decade) and 

