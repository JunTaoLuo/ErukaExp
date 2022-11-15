'''
Code to import clean building_info data, and reshape it to wide
'''

import pandas as pd

def reshape_building_data(df):
  '''
  Input: building data as pd dataframe, read from processed schema
  Output: dataframe reshaped from long to wide, with unique parcelid
  '''

  # Separate out columns that are constant
  fixed = df[['parcelid', 'bsmt_sqft', 'prop_class_code', 'class_description', 'tot_buildings']]
  fixed = fixed.drop_duplicates('parcelid')

  to_pivot = df.drop(['bsmt_sqft', 'prop_class_code', 'class_description', 'tot_buildings'], axis=1)

  result = to_pivot.pivot(index='parcelid', columns='building_id')
  result.columns = result.columns.map(lambda x: '{}_{}'.format(x[0], x[1]))
  result.reset_index()

  final = pd.merge(result, fixed, on='parcelid')

  return final