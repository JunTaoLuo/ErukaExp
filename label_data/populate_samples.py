import argparse
import os
import sys
from sqlalchemy import create_engine
from jinja2 import Environment, FileSystemLoader
import constants

def initialize_tables(connection, env, params, verbose):
    print("Initializing tables")

    template = env.get_template("initialize_tables.sql.j2")
    query = template.render(params)

    if verbose:
        print(query)

    connection.execute(query)

    print(".. done")

def populate_samples(connection, env, params, verbose):
    print("Populating samples")

    template = env.get_template("populate_samples.sql.j2")
    query = template.render(params)

    if verbose:
        print(query)

    connection.execute(query)

    print(".. done")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Update sample labels",
        description="Script for uploading labeling results from csv file")
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output, including executed SQL queries')
    args = parser.parse_args()

    if 'ERUKA_DB' not in os.environ or not os.environ['ERUKA_DB']:
        print('No PostgreSQL endpoing configured, please specify connection string via ERUKA_DB environment variable')
        sys.exit()

    eruka_db_str = os.environ['ERUKA_DB']

    db = create_engine(eruka_db_str)
    jinja_env = Environment(loader=FileSystemLoader(constants.template_dir))

    with db.connect() as conn:
        initialize_tables(conn, jinja_env, constants.db_params, args.verbose)
        populate_samples(conn, jinja_env, constants.db_params, args.verbose)