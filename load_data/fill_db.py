import pandas as pd
import os.path
from sqlalchemy import create_engine
from pathlib import Path
import ohio.ext.pandas # library with better pandas -> postgreSQL writing

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# TODO: update db connection string
db_str = # Enter connection string
db = create_engine(db_str)

def populate_db_from_file(drive: GoogleDrive, filepath: str, file_id: str, db, table_name: str):
    print(f"Downloading file: {filepath}")
    file = drive.CreateFile({"id": file_id})
    file.GetContentFile(filepath)

    print(f"Parsing {table_name}:")
    file_extension = Path(filepath).suffix
    if file_extension == ".csv":
        file_df = pd.read_csv(filepath)
    elif file_extension == ".xlsx":
        file_df = pd.read_excel(filepath)
    else:
        raise Exception(f"Unrecognized file extension for file {filepath}")

    print("Writing to Postgres")
    file_df.pg_copy_to(table_name, db, schema = 'raw', if_exists="replace", index=False)

    print("Done")

if not os.path.exists("Dataset"):
    os.makedirs("Dataset")

ga = GoogleAuth()
ga.LocalWebserverAuth()  # This line in your code currently calls LocalWebserverAuth()
drive = GoogleDrive(ga)

files = [
    ("./Dataset/buildingInfo.xlsx", "1iRLn6YTeOWRZUJosOW80Fi4QGyzBDror", "BuildingInfo"),
    ("./Dataset/historicSales.xlsx", "1VvRONG_HyYi5gJEIf4gUyWxRIaabThWw", "HistoricSales"),
    ("./Dataset/monthlyTax.xlsx", "18dvcSrKSEFQFYLmMcBxUv6vnoifXm0bz", "MonthlyTax"),
    ("./Dataset/rentalRegistration.csv", "1hM7i_Wjds4fLZQTer3cRHh6RsmDFR2FR", "RentalRegistration"),
    ("./Dataset/propertyTransfer.csv", "157d-wQZdt3WxM1H32Ihw3TMsn8bsPBYb", "PropertyTransfer")
]

for (filepath, file_id, table_name) in files:
    populate_db_from_file(drive, filepath, file_id, db, table_name)
