
import argparse
import random
import constants
import csv
import os
import requests
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script for verifying results from csv file")
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output, including executed SQL queries')
    parser.add_argument('-g', '--gdrive', nargs=1, required=True, help='indicate the GDrive folder containing the results')
    args = parser.parse_args()

    print(f"Getting results from GDrive at {args.gdrive}")

    # Create clean directory if not exists
    if not os.path.exists(constants.data_dir):
        os.makedirs(constants.data_dir)
    for f in os.listdir(constants.data_dir):
        os.remove(os.path.join(constants.data_dir, f))

    # ErukaLabels folder id
    labels_dir_id = "1WJ50iIVfCKRFPYOjBPvCYaSke8JOiLWz"

    # Ref csv file id
    ref_csv_id = "1n5JaxpzEDCACnFlyUPnCJwquiMCxnTAY"

    # Sign in to GDrive
    ga = GoogleAuth()
    ga.LocalWebserverAuth()  # This line in your code currently calls LocalWebserverAuth()
    drive = GoogleDrive(ga)

    # Download ref csv
    ref_csv_file = drive.CreateFile({'id': ref_csv_id})
    ref_csv_file.GetContentFile(constants.ref_csv_file)

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
    incorrect_parcelids = []
    abs_error = 0

    # Check csv file for syntax
    with open(constants.ref_csv_file, "r") as ref_file, open(constants.building_labels_file, "r") as labels_file:
        ref_labels = csv.DictReader(ref_file)
        labels = csv.DictReader(labels_file)

        for line, (ref_row, labels_row) in enumerate(zip(ref_labels, labels)):
            parcelid = ref_row["parcelid"]
            num_labels += 1
            label_str = labels_row['building_value'].replace(',', '')

            if (ref_row['building_value'] != label_str
                or ref_row['year'] != labels_row['year']
                or ref_row['handwritten'] != labels_row['handwritten']):
                print(f"Error in labeling {parcelid}")

                if ref_row['building_value'] != label_str:
                    abs_error += abs(int(ref_row['building_value']) - int(label_str))
                    print(f"building value incorrect expected: {ref_row['building_value']} labeled: {labels_row['building_value']}")
                if ref_row['year'] != labels_row['year']:
                    print(f"year incorrect expected: {ref_row['year']} labeled: {labels_row['year']}")
                if ref_row['handwritten'] != labels_row['handwritten']:
                    print(f"handwritten incorrect expected: {ref_row['handwritten']} labeled: {labels_row['handwritten']}")

                incorrect_parcelids.append(parcelid)

    print(f"Labeling Accuracy: {1 - (len(incorrect_parcelids)/num_labels)}")
    print("Incorrect parcels:")
    for id in incorrect_parcelids:
        print(id)
    print(f"Building Value MAE: {abs_error/num_labels}")
