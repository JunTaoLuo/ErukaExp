import argparse
import os
import sys
from sqlalchemy import create_engine, text
from jinja2 import Environment, FileSystemLoader
import constants

def initialize_tables(connection, env, params, verbose):
    print("Initializing tables")

    template = env.get_template("initialize_tables.sql.j2")
    query = template.render(params)

    if verbose:
        print(query)

    connection.execute(text(query))
    connection.commit()

    print(".. done")

def populate_samples(connection, env, params, verbose):
    print("Populating samples")

    template = env.get_template("populate_samples.sql.j2")
    query = template.render(params)

    if verbose:
        print(query)

    connection.execute(text(query))
    connection.commit()

    print(".. done")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script for populating empty samples labels")
    parser.add_argument('-s', '--schema', required=True, choices=['hamilton', 'franklin'], help='verbose output, including executed SQL queries')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output, including executed SQL queries')
    args = parser.parse_args()

    if 'ERUKA_DB' not in os.environ or not os.environ['ERUKA_DB']:
        print('No PostgreSQL endpoing configured, please specify connection string via ERUKA_DB environment variable')
        sys.exit()

    eruka_db_str = os.environ['ERUKA_DB']

    db = create_engine(eruka_db_str)
    jinja_env = Environment(loader=FileSystemLoader(constants.template_dir))


    template_params = dict(constants.db_params)
    if args.schema == "hamilton":
        template_params['schema'] = constants.db_params["hamilton_schema"]
    elif args.schema == "franklin":
        template_params['schema'] = constants.db_params["franklin_schema"]

    with db.connect() as conn:
        initialize_tables(conn, jinja_env, template_params, args.verbose)
        populate_samples(conn, jinja_env, template_params, args.verbose)