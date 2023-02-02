import argparse
import constants
import csv
import os
import sys
import requests
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script for uploading labeling results from csv file")
    parser.add_argument('-c', '--csv', help='an input csv file of the parcelids to retrieve')
    parser.add_argument('-g', '--gdrive', nargs=1, required=False, help='indicate the GDrive folder containing the results',)
    args = parser.parse_args()

    if len(args.gdrive) > 0 and args.gdrive[0]:
        print(f"Getting results from GDrive at {args.gdrive}")

        # ErukaLabels folder id
        labels_dir_id = "1WJ50iIVfCKRFPYOjBPvCYaSke8JOiLWz"

        # Sign in to GDrive
        ga = GoogleAuth()
        ga.LocalWebserverAuth()  # This line in your code currently calls LocalWebserverAuth()
        drive = GoogleDrive(ga)

        # Get all expected files from data directory
        parcelids = []
        with open(args.csv, "r") as f:
            csv = csv.reader(f)
            for row in csv:
                parcelids.append(f"{row[0]}.jpg")
            parcelids.sort()
        print(f"Verifying existence of {len(parcelids)} files")

        folder_id = ""
        file_list = drive.ListFile({'q': f"'{labels_dir_id}' in parents and trashed=false"}).GetList()
        for file in file_list:
            if(file['title'] == args.gdrive[0]):
                folder_id = file['id']
                break

        if not folder_id:
            print(f"Could not find folder {args.gdrive} on GDrive")
            exit(1)

        file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
        missing = 0
        for file in parcelids:
            if not any(f for f in file_list if f['title'] == file):
                print(f"Missing file: {file}")
                missing += 1
        print(f"Total missing files: {missing}")
        print("Verification complete")

