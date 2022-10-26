import csv
from sqlalchemy import create_engine

# TODO: update db connection string
db_str = # connection string
db = create_engine(db_str)

update_query = """
update ocrtargets
set {0} = {1}
where parcel_number = '{2}';"""


with open("Dataset/Ownership/buildings.csv", "r") as f:
    building_labels = csv.DictReader(f)

    for row in building_labels:
        parcel = row["parcel"]
        building_value = row["value"]

        if building_value:
            query = update_query.format("initial_building_value", building_value, parcel)
            print(f"Executing {query}")
            db.execute(query)

with open("Dataset/Ownership/land.csv", "r") as f:
    land_labels = csv.DictReader(f)

    for row in land_labels:
        parcel = row["parcel"]
        land_value = row["value"]

        if land_value:
            query = update_query.format("initial_land_value", land_value, parcel)
            print(f"Executing {query}")
            db.execute(query)
