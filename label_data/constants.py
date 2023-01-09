import os

# Directories

script_dir = os.path.dirname(__file__)
template_dir = os.path.join(script_dir, "sql_templates")
data_dir = os.path.join(script_dir, "data")

# Files
building_labels_prefix = "building_labels"
building_labels_name = f"{building_labels_prefix}.csv"
building_labels_file = os.path.join(data_dir, building_labels_name)
ref_csv_file = os.path.join(data_dir, "ref.csv")

# Parameters

db_params = {
    "samples_label_table": "samples.labels",
    "samples_building_values_table": "samples.building_values",
    "samples_download_time_table": "samples.download_time",
    "samples_label_time_table": "samples.label_time",
    "samples_error_table": "samples.error",
    "building_info_source": "processed.building_info",
    "historic_sales_source": "processed.historic_sales"
}