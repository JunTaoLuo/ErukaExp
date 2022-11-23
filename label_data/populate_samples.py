import os
import sys
from sqlalchemy import create_engine
from jinja2 import Environment, FileSystemLoader
import constants

script_dir = os.path.dirname(__file__)
template_dir = os.path.join(script_dir, "sql_templates")

def initialize_tables(connection, env, params):
    print("Initializing tables")

    template = env.get_template("initialize_tables.sql.j2")
    query = template.render(params)

    connection.execute(query)

    print(".. done")

def populate_samples(connection, env, params):
    print("Populating samples")

    template = env.get_template("populate_samples.sql.j2")
    query = template.render(params)

    print(query)

    connection.execute(query)
    # result = connection.execute(query).fetchall()

    # print(f"Inserted {len(result)} samples")

    print(".. done")

if __name__ == "__main__":

    if 'ERUKA_DB' not in os.environ:
        print('No PostgreSQL endpoing configured, please specify connection string via ERUKA_DB environment variable')
        sys.exit()

    eruka_db_str = os.environ['ERUKA_DB']

    db = create_engine(eruka_db_str)
    jinja_env = Environment(loader=FileSystemLoader(template_dir))

    with db.connect() as conn:
        initialize_tables(conn, jinja_env, constants.db_params)
        populate_samples(conn, jinja_env, constants.db_params)