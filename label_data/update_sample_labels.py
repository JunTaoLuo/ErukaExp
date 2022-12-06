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

    num_labels = 0

    # Check csv file for syntax
    with open(constants.building_labels_file, "r") as f:
        building_labels = csv.DictReader(f)

        for line, row in enumerate(building_labels):
            num_labels += 1
            parcelid = row["parcelid"]
            value_no_year = row["value_no_year"]

            # value_no_year must be empty, numeric or "error"
            if value_no_year:
                if value_no_year != "error" and not value_no_year.isnumeric():
                    print(f"Invalid value for 'value_no_year': {value_no_year} for parcel {parcelid} on line {line+2}. Value must be empty, numeric or 'error'.")
                    sys.exit()

            # year1 and value1 must both be empty or numeric
            year1 = row["year1"]
            value1 = row["value1"]

            if (year1 and not value1) or (not year1 and value1):
                print(f"Mismatched 'year1': {year1} and 'value1': {value1} for parcel {parcelid} on line {line+2}. Both must be empty or non-empty.")
                sys.exit()
            if year1 and not year1.isnumeric():
                print(f"Invalid value for 'year1': {year1} for parcel {parcelid} on line {line+2}. Value must be numeric.")
                sys.exit()
            if value1 and not value1.isnumeric():
                print(f"Invalid value for 'value1': {value1} for parcel {parcelid} on line {line+2}. Value must be numeric.")
                sys.exit()

            # year2 and value2 must both be empty or numeric
            year2 = row["year2"]
            value2 = row["value2"]

            if (year2 and not value2) or (not year2 and value2):
                print(f"Mismatched 'year2': {year2} and 'value2': {value2} for parcel {parcelid} on line {line+2}. Both must be empty or non-empty.")
                sys.exit()
            if year2 and not year2.isnumeric():
                print(f"Invalid value for 'year2': {year2} for parcel {parcelid} on line {line+2}. Value must be numeric.")
                sys.exit()
            if value2 and not value2.isnumeric():
                print(f"Invalid value for 'value2': {value2} for parcel {parcelid} on line {line+2}. Value must be numeric.")
                sys.exit()

            # if there is an error, none of year1, value1, year2, value2 should be set
            if value_no_year == "error" and (year1 or value1 or year2 or value2):
                print(f"Invalid entry for parcel {parcelid} on line {line+2}. No year or value should be entered if error has occured.")
                sys.exit()

            # year2, value2 should only be set if year1, value1 is set
            if (year2 or value2) and not year1 and not value1:
                print(f"Invalid entry for parcel {parcelid} on line {line+2}. 'year2' and 'value2' set while 'year1' and 'value1' not set.")
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
    template = jinja_env.get_template("update_labels.sql.j2")

    with db.connect() as conn, open(constants.building_labels_file, "r") as f:
        building_labels = csv.DictReader(f)
        print(f"Uploading {num_labels} labels")

        for i, row in enumerate(building_labels):
            print(f"Uploading label {i}")
            params = dict(constants.db_params)
            params["parcelid"] = row["parcelid"]
            value_no_year = row["value_no_year"]
            year1 = row["year1"]
            value1 = row["value1"]
            year2 = row["year2"]
            value2 = row["value2"]
            params["update"] = value_no_year or year1 or value1 or year2 or value2
            params["error"] = value_no_year == "error"
            params["value_no_year"] = value_no_year
            params["year1"] = year1
            params["value1"] = value1
            params["year2"] = year2
            params["value2"] = value2

            query = template.render(params)

            if args.verbose:
                print(query)

            conn.execute(query)

