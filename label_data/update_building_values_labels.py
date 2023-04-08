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
    parser.add_argument('-f', '--file', required=False, default=constants.building_values_prefix, help='indicate the result file by name in the specified folder')
    parser.add_argument('-g', '--gdrivefolder', nargs=1, required=False, help='indicate the GDrive folder by name containing the results')
    parser.add_argument('-i', '--gdriveid', nargs=1, required=False, help='indicate the GDrive folder by id containing the results')
    parser.add_argument('-s', '--schema', required=True, choices=['hamilton', 'franklin'], help='which schema to use, hamilton or franklin')
    args = parser.parse_args()

    result_path = os.path.join(constants.data_dir, f"{args.file}.csv")

    if (args.gdrivefolder and args.gdrivefolder[0]) or (args.gdriveid and args.gdriveid[0]):
        print(f"Getting results from GDrive at {args.gdrivefolder}|{args.gdriveid}")

        # ErukaLabels folder id
        labels_dir_id = "1WJ50iIVfCKRFPYOjBPvCYaSke8JOiLWz"

        # Sign in to GDrive
        ga = GoogleAuth()
        ga.LocalWebserverAuth()  # This line in your code currently calls LocalWebserverAuth()
        drive = GoogleDrive(ga)

        folder_id = ""

        if args.gdrivefolder and args.gdrivefolder[0]:
            file_list = drive.ListFile({'q': f"'{labels_dir_id}' in parents and trashed=false"}).GetList()
            for file in file_list:
                if(file['title'] == args.gdrivefolder[0]):
                    folder_id = file['id']
                    break
            if not folder_id:
                print(f"Could not find folder {args.gdrivefolder} on GDrive")
                exit(1)
        elif args.gdriveid and args.gdriveid[0]:
            folder_id = args.gdriveid[0]
        else:
            print(f"Must provide GDrive folder by name or id")
            exit(1)

        export_url = ""
        file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
        for file in file_list:
            if(file['title'] == args.file):
                export_url = file['exportLinks']['text/csv']
                break

        if not export_url:
            print(f"Could not find file {args.file} in {args.gdrivefolder} on GDrive")
            exit(1)

        headers = {'Authorization': 'Bearer ' + ga.credentials.access_token}
        res = requests.get(export_url, headers=headers)
        open(result_path, "wb").write(res.content)

    num_labels = 0

    # Check csv file for syntax
    with open(result_path, "r") as f:
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

    if not os.path.exists(result_path):
        print(f'Results file: {result_path} not found')
        sys.exit()

    eruka_db_str = os.environ['ERUKA_DB']

    db = create_engine(eruka_db_str)
    jinja_env = Environment(loader=FileSystemLoader(constants.template_dir))
    template = jinja_env.get_template("update_building_values_labels.sql.j2")

    with db.connect() as conn, open(result_path, "r") as f:
        building_labels = csv.DictReader(f)
        print(f"Uploading {num_labels} labels")

        for i, row in enumerate(building_labels):
            print(f"Uploading label {i}")
            params = dict(constants.db_params)
            params["building_values_table"] = args.file
            params["schema"] = constants.db_params["hamilton_schema"] if args.schema == "hamilton" else constants.db_params["franklin_schema"]
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

