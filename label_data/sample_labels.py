import argparse
import random
import constants
import csv
import os
import sys
import requests
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script for sampling results from csv file")
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output, including executed SQL queries')
    parser.add_argument('-g', '--gdrive', nargs=1, required=True, help='indicate the GDrive folder containing the results')
    parser.add_argument('-c', '--count', nargs=1, type=int, required=False, default=100, help='indicate the number of results to sasmple')
    args = parser.parse_args()

    print(f"Getting results from GDrive at {args.gdrive}")

    # Create clean directory if not exists
    if not os.path.exists(constants.data_dir):
        os.makedirs(constants.data_dir)
    for f in os.listdir(constants.data_dir):
        os.remove(os.path.join(constants.data_dir, f))

    # ErukaLabels folder id
    labels_dir_id = "1WJ50iIVfCKRFPYOjBPvCYaSke8JOiLWz"

    # Sign in to GDrive
    ga = GoogleAuth()
    ga.LocalWebserverAuth()  # This line in your code currently calls LocalWebserverAuth()
    drive = GoogleDrive(ga)

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

    results_file = os.path.join(constants.data_dir, "all_labels.csv")
    results_file = os.path.join(constants.data_dir, "all_labels.csv")

    open(results_file, "wb").write(res.content)

    num_labels = 0
    with open(results_file) as f:
        num_labels = sum(1 for _ in f)

    # Sample N rows among labels
    samples = list(range(num_labels))
    random.shuffle(samples)
    samples = samples[:args.count]
    samples.sort()

    # Check csv file for syntax
    with open(results_file, "r") as f, open(constants.building_labels_file, "w") as fout:
        building_labels = csv.DictReader(f)
        index = 0
        fout.write(f"parcelid,building_value,year,handwritten\n")

        for line, row in enumerate(building_labels):
            if index >= args.count:
                break
            if line != samples[index]:
                continue
            else:
                index += 1

            parcelid = row["parcelid"]
            fout.write(f"'{parcelid},{row['building_value']},{row['year']},{row['handwritten']}\n")
            image_file = [file for file in file_list if file['title'] == f'{parcelid}.jpg']
            image_file[0].GetContentFile(os.path.join(constants.data_dir, f'{parcelid}.jpg'))

    os.remove(results_file)
