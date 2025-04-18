### This script uploads our clean structured datasets to Hugging Face ###

from huggingface_hub import HfApi
import os
import pandas as pd
from sqlalchemy import create_engine
import yaml
import sshtunnel


# Export the HF_TOKEN environment variable using python

def db_connect(secrets):
    """
    Connects to a PostgreSQL database via SSH tunnel if enabled.
    """
    db_params = secrets['db']

    tunnel = None

    if db_params.get('use_tunnel', False):
        print('Using SSH tunnel...')
        
        tunnel = sshtunnel.SSHTunnelForwarder(
            (db_params['ssh_host'], db_params.get('ssh_port', 22)),  # Use default SSH port 22 if not provided
            ssh_username=db_params.get('ssh_user', ''),
            ssh_pkey=db_params['ssh_pkey_path'],
            ssh_private_key_password=db_params.get('ssh_pkey_pass', None),
            remote_bind_address=('127.0.0.1', db_params['port']),  # Ensure PostgreSQL is bound correctly
            local_bind_address=('localhost', db_params['local_port'])
        )

        tunnel.start()

        engine = create_engine(
            f"postgresql+psycopg2://{db_params['user']}:{db_params['pass']}@localhost:{db_params['local_port']}/{db_params['dbname']}?sslmode=allow"
        )

    else:
        print('Connecting directly...')
        engine = create_engine(
            f"postgresql://{db_params['user']}:{db_params['pass']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
        )

    return engine, tunnel

def upload_folder(folder_path):
    """
    Uploads the dataset to Hugging Face.
    """
    # Initialize the API client
    # Make sure to set the HF_TOKEN environment variable        
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_folder(
        folder_path=folder_path,
        repo_id="eruka-cmu-housing/historical-appraisals-ocr-ml",
        repo_type="dataset"
    )

if __name__ == '__main__':

    # Connect to DB
    with open('database.yaml', 'r') as f:
        secrets = yaml.safe_load(f)

    eng, tunnel = db_connect(secrets)

    # Pull datasets and write them to csv files in local directory
    local_dir = "huggingface_data"
    with eng.connect() as conn:

        # Hamilton county features
        hamilton_feats_query = '''
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
         '''

        hamilton_feats = pd.read_sql(hamilton_feats_query, eng)
        print(f"Hamilton county features shape: {hamilton_feats.shape}")
        hamilton_feats.to_csv(os.path.join(local_dir,'hamilton_county','building_features_hamilton.csv'), index=False)

        # Hamilton county hand_labeled
        sql = '''SELECT parcelid, building_value,
                        COALESCE(year, 1933) as "year",
                        case when handwritten is True then 1 else 0 end as "appraisal_handwritten_flag"
                 FROM samples.building_values WHERE building_value is not null
              '''
        hamilton_handlabeled = pd.read_sql(sql, eng)
        print(f"Hamilton county hand_labeled shape: {hamilton_handlabeled.shape}")
        hamilton_handlabeled = hamilton_handlabeled[hamilton_handlabeled['year'] == 1933]
        hamilton_handlabeled.to_csv(os.path.join(local_dir,'hamilton_county','hand_annotations_single_cell_hamilton.csv'), index=False)

        # Franklin county features

        franklin_feats = pd.read_sql('SELECT * FROM processed.franklin_features', eng)
        print(f"Franklin county features shape: {franklin_feats.shape}")
        franklin_feats.to_csv(os.path.join(local_dir,'franklin_county','building_features_franklin.csv'), index=False)

        # Franklin county hand_labeled
        franklin_labels_1920 = pd.read_sql('''select parcelid, building_value, year
                                          from franklin.building_values
                                          where year = 20 and building_value is not null''', eng)

        franklin_labels_1931 = pd.read_sql('''select parcelid, building_value
                                          from franklin.building_values_1930
                                          where building_value is not null''', eng)

        franklin_labels_1931['year'] = 1931
        franklin_labels_1920['year'] = 1920

        franklin_labels = pd.concat([franklin_labels_1920, franklin_labels_1931], axis=0)
        franklin_labels.to_csv(os.path.join(local_dir,'franklin_county','hand_annotations_single_cell_franklin.csv'), index=False)

        # Write the entire huggingface_data directory (not including the huggingface_data level) to my huggin face repo
        upload_folder(local_dir)


