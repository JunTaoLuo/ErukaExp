import os

# Directories

script_dir = os.path.dirname(__file__)
template_dir = os.path.join(script_dir, "sql_templates")
data_dir = os.path.join(script_dir, "data")
samples_dir = os.path.join(script_dir, "samples")

# Files
building_values_prefix = "building_values"
building_values_name = f"{building_values_prefix}.csv"
building_values_file = os.path.join(data_dir, building_values_name)
ref_csv_file = os.path.join(data_dir, "ref.csv")

# Parameters

db_params = {
    "franklin_schema": "franklin",
    "hamilton_schema": "samples",
    "label_table": "labels",
    "building_values_table": "building_values",
    "download_time_table": "download_time",
    "label_time_table": "label_time",
    "error_table": "error",
    "building_info_source": "processed.building_info",
    "historic_sales_source": "processed.historic_sales"
}