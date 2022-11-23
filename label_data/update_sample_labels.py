import argparse
import constants
import csv
import os
import sys
from sqlalchemy import create_engine
from jinja2 import Environment, FileSystemLoader

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script for uploading labeling results from csv file")
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output, including executed SQL queries')
    args = parser.parse_args()

    if 'ERUKA_DB' not in os.environ or not os.environ['ERUKA_DB']:
        print('No PostgreSQL endpoing configured, please specify connection string via ERUKA_DB environment variable')
        sys.exit()

    if not os.path.exists(constants.building_labels_file):
        print(f'Results file: {constants.building_labels_file} not found')
        sys.exit()

    eruka_db_str = os.environ['ERUKA_DB']

    db = create_engine(eruka_db_str)
    jinja_env = Environment(loader=FileSystemLoader(constants.template_dir))
    template = jinja_env.get_template("update_labels.sql.j2")

    with db.connect() as conn, open(constants.building_labels_file, "r") as f:

        building_labels = csv.DictReader(f)

        for row in building_labels:
            params = dict(constants.db_params)
            params["parcelid"] = row["parcelid"]
            initial_building_value = row["initial_building_value"]
            if initial_building_value:
                if initial_building_value == "error":
                    params["error"] = True
                else:
                    params["error"] = False
                    params["initial_building_value"] = initial_building_value

            query = template.render(params)

            if args.verbose:
                print(query)

            conn.execute(query)

