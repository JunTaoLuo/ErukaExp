import argparse
import constants
import csv
import os
import sys
import requests
from sqlalchemy import create_engine
from jinja2 import Environment, FileSystemLoader
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script for uploading labeling results from csv file")
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output, including executed SQL queries')
    parser.add_argument('-g', '--gdrive', nargs=1, required=False, help='indicate the GDrive folder containing the results',)
    args = parser.parse_args()

    if len(args.gdrive) > 0 and args.gdrive[0]:
        print(f"Getting results from GDrive at {args.gdrive}")

        # ErukaLabels folder id
        labels_dir_id = "1WJ50iIVfCKRFPYOjBPvCYaSke8JOiLWz"

        # Sign in to GDrive
        ga = GoogleAuth()
        ga.LocalWebserverAuth()  # This line in your code currently calls LocalWebserverAuth()
        drive = GoogleDrive(ga)

        folder_id = ""
        file_list = drive.ListFile({'q': f"'{labels_dir_id}' in parents and trashed=false"}).GetList()
        for file in file_list:
            if(file['title'] == args.gdrive[0]):
                folder_id = file['id']
                break

        if not folder_id:
            print(f"Could not find folder {args.gdrive} on GDrive")
            exit(1)

        export_url = ""
        file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
        for file in file_list:
            if(file['title'] == constants.building_labels_prefix):
                export_url = file['exportLinks']['text/csv']
                break

        if not export_url:
            print(f"Could not find file {constants.building_labels_prefix} in {args.gdrive} on GDrive")
            exit(1)

        headers = {'Authorization': 'Bearer ' + ga.credentials.access_token}
        res = requests.get(export_url, headers=headers)
        open(constants.building_labels_file, "wb").write(res.content)

    num_labels = 0

    # Check csv file for syntax
    with open(constants.building_labels_file, "r") as f:
        building_labels = csv.DictReader(f)

        for line, row in enumerate(building_labels):
            num_labels += 1
            parcelid = row["parcelid"]
            building_value = row["building_value"].lower()

            # building_value must be empty, numeric or "error"
            if building_value:
                if building_value != "error" and not building_value.isnumeric():
                    print(f"Invalid value for 'building_value': {building_value} for parcel {parcelid} on line {line+2}. Value must be empty, numeric or 'error'.")
                    sys.exit()

            # year must be empty or numeric and of length 2
            year = row["year"]

            if year:
                if not year.isnumeric():
                    print(f"Invalid value for 'year': {year} for parcel {parcelid} on line {line+2}. Value must be numeric.")
                    sys.exit()
                if len(year) != 2:
                    print(f"Invalid value for 'year': {year} for parcel {parcelid} on line {line+2}. Value must consist of 2 digits.")
                    sys.exit()

            # handwritten be empty or '1'
            handwritten = row["handwritten"]

            if handwritten and handwritten != "1":
                    print(f"Invalid value for 'handwritten': {handwritten} for parcel {parcelid} on line {line+2}. Value must be empty or '1'.")
                    sys.exit()

    if 'ERUKA_DB' not in os.environ or not os.environ['ERUKA_DB']:
        print('No PostgreSQL endpoing configured, please specify connection string via ERUKA_DB environment variable')
        sys.exit()

    if not os.path.exists(constants.building_labels_file):
        print(f'Results file: {constants.building_labels_file} not found')
        sys.exit()

    eruka_db_str = os.environ['ERUKA_DB']

    db = create_engine(eruka_db_str)
    jinja_env = Environment(loader=FileSystemLoader(constants.template_dir))
    template = jinja_env.get_template("update_building_values_labels.sql.j2")

    with db.connect() as conn, open(constants.building_labels_file, "r") as f:
        building_labels = csv.DictReader(f)
        print(f"Uploading {num_labels} labels")

        for i, row in enumerate(building_labels):
            print(f"Uploading label {i}")
            params = dict(constants.db_params)
            params["parcelid"] = row["parcelid"]
            building_value = row["building_value"].lower()
            year = row["year"]
            handwritten = row["handwritten"] == "1"
            params["update"] = building_value or year or handwritten
            params["error"] = building_value == "error"
            params["building_value"] = building_value
            params["year"] = year
            params["handwritten"] = handwritten

            query = template.render(params)

            if args.verbose:
                print(query)

            conn.execute(query)

