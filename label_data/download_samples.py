import argparse
import csv
import constants
import cv2
import imutils
import numpy as np
import os
import pdf2image as p2i
from pytesseract import image_to_osd, Output
import re
import requests as rq
import sys
from sqlalchemy import create_engine, text
from jinja2 import Environment, FileSystemLoader
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from datetime import datetime

# Ownership page source template
ownership_source = 'https://wedge.hcauditor.org/view/re/{}/2021/imagesOwnerCard'
ownership_reference_regex = '.*convert\/(.*.tif)\/.*'
ownership_file = 'https://wedge.hcauditor.org/convert/{}/'
franklin_source = 'https://audr-apps.franklincountyohio.gov/ParcelSheet/{}'

def download_franklin_file(parcelid, verbose):
    print(f"Processing {parcelid}")

    # Download PDF
    ownership_file_formated = franklin_source.format(parcelid)
    print(f"Downloading ownership card PDF {ownership_file_formated}")

    response = rq.get(ownership_file_formated)
    if not response.ok:
        print(f"Failed to get {ownership_file_formated}")
        return "PDF download failure"

    # Convert to jpg
    try:
        pages = p2i.convert_from_bytes(response.content)
    except:
        return "JPEG conversion failure"

    if len(pages) == 0:
        print("Failed to convert PDF to JPEG")
        return "JPEG conversion failure"

    jpeg_file = os.path.join(constants.data_dir, f"{parcelid}.jpg")
    if os.path.exists(jpeg_file):
        os.remove(jpeg_file)
    pages[0].save(jpeg_file, "JPEG")

    return "Success"


def download_hamilton_file(parcelid, verbose):
    print(f"Processing {parcelid}")

    # Retrieve ownership card page source (HTML)
    ownership_source_formatted = ownership_source.format(parcelid)
    print(f"Downloading source {ownership_source_formatted}")

    response = rq.get(ownership_source_formatted)
    if not response.ok:
        print(f"Failed to get {ownership_source_formatted}")
        return "Source download failure"

    # Search for ownership card PDF file reference
    # print(response.text)
    oc_reference = ''
    lines = response.text.splitlines()

    for line in lines:
        if "var selected" in line:
            print(f"Checking for reference in: {line}")
            match = re.search(ownership_reference_regex, line)
            if match:
                oc_reference = match.group(1)

    if not oc_reference:
        if verbose:
            print(f"Failed to find ownership card reference in {response.text}")
        else:
            print(f"Failed to find ownership card reference")
        return "PDF reference not found"

    # Download PDF
    ownership_file_formated = ownership_file.format(oc_reference)
    print(f"Downloading ownership card PDF {ownership_file_formated}")

    response = rq.get(ownership_file_formated)
    if not response.ok:
        print(f"Failed to get {ownership_file_formated}")
        return "PDF download failure"

    # Convert to jpg
    pages = p2i.convert_from_bytes(response.content)

    if len(pages) == 0:
        print("Failed to convert PDF to JPEG")
        return "JPEG conversion failure"

    if len(pages) == 1:
        jpeg_file = os.path.join(constants.data_dir, f"{parcelid}.jpg")
        if os.path.exists(jpeg_file):
            os.remove(jpeg_file)
        pages[0].save(jpeg_file, "JPEG")
    else:
        for i, page in enumerate(pages):
            jpeg_file = os.path.join(constants.data_dir, f"{parcelid}-{i}.jpg")
            if os.path.exists(jpeg_file):
                os.remove(jpeg_file)
            page.save(jpeg_file, "JPEG")

    return "Success"

def crop_image(img):
    # Crops the whitespace in the image
    # Code sourced from: https://stackoverflow.com/questions/49907382/how-to-remove-whitespace-from-an-image-in-opencv
    # Seems to work well based on manual tests on the images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # grayscale the image
    gray = 255*(gray < 128).astype(np.uint8) # To invert the text to white
    coords = cv2.findNonZero(gray) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box that covers all text
    rect = img[y:y+h, x:x+w] # Crop the image - note we do this on the original image

    return rect

def fix_orientation(img):
    # Fetches info about the orientation, rotation needed to fix the image
    results = image_to_osd(img, output_type=Output.DICT)

    if results['orientation'] == 0: # if image is already in correct orientation, do nothing
        return img
    else:
        return imutils.rotate_bound(img, angle=results["rotate"]) # rotate by the amount identified by pytesseract

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script for downloading new samples for manual labeling")
    parser.add_argument('-c', '--csv', help='an input csv file of the parcelids to retrieve')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output, including executed SQL queries')
    parser.add_argument('-su', '--skip-upload', action='store_true', help='do not upload to google drive')
    parser.add_argument('-p', '--parcelid', help='download sample for specific parcel ID, this option overrides N and does not record the sample as downloaded')
    parser.add_argument('-s', '--schema', required=True, choices=['hamilton', 'franklin'], help='which schema to use, hamilton or franklin')
    parser.add_argument('entries', metavar="N", type=int, help="Number of entries to retrieve", default=20, nargs='?')
    args = parser.parse_args()

    if 'ERUKA_DB' not in os.environ or not os.environ['ERUKA_DB']:
        print('No PostgreSQL endpoing configured, please specify connection string via ERUKA_DB environment variable')
        sys.exit()

    if not args.skip_upload:
        # Sign in to GDrive
        ga = GoogleAuth()
        ga.LocalWebserverAuth()  # This line in your code currently calls LocalWebserverAuth()
        drive = GoogleDrive(ga)

    # Create clean directory if not exists
    if not os.path.exists(constants.data_dir):
        os.makedirs(constants.data_dir)
    for f in os.listdir(constants.data_dir):
        os.remove(os.path.join(constants.data_dir, f))

    eruka_db_str = os.environ['ERUKA_DB']

    db = create_engine(eruka_db_str)
    jinja_env = Environment(loader=FileSystemLoader(constants.template_dir))
    template = jinja_env.get_template("retrieve_new_samples.sql.j2")
    parcelids = []

    params = dict(constants.db_params)
    params["entries"] = args.entries
    params["schema"] = args.schema

    query = template.render(params)

    with db.connect() as conn:
        if args.parcelid:
            parcelids.append(args.parcelid)
        elif args.csv:
            with open(args.csv, "r") as f:
                csv = csv.reader(f)
                for row in csv:
                    parcelids.append(row[0])
            parcelids.sort() # sorting to ensure they appear in order on labels csv file
        else:
            if args.verbose:
                print(query)

            # Get list of parcelids from database
            results = conn.execute(text(query)).fetchall()
            conn.commit()
            for row in results:
                parcelids.append(row[0])
            parcelids.sort() # sorting to ensure they appear in order on labels csv file

        # Put all non-numeric parcelids first to match file explorer/finder
        parcelids_numeric = []
        parcelids_nonnumeric = []
        for parcelid in parcelids:
            if parcelid.isnumeric():
                parcelids_numeric.append(parcelid)
            else:
                parcelids_nonnumeric.append(parcelid)
        parcelids = parcelids_nonnumeric
        parcelids.extend(parcelids_numeric)

        template = jinja_env.get_template("update_errors.sql.j2")

        downloaded_parcelids = []

        # Download OCs
        for parcelid in parcelids:
            if args.schema == "hamilton":
                status = download_hamilton_file(parcelid, args.verbose)
            elif args.schema == "franklin":
                status = download_franklin_file(parcelid, args.verbose)

            if status != "Success":
                params = dict(constants.db_params)
                params["schema"] = args.schema
                params["parcelid"] = parcelid
                params["error"] = status
                query = template.render(params)
                conn.execute(text(query))
                conn.commit()
                print(f"Error: {status}")
            else:
                downloaded_parcelids.append(parcelid)

        # Write labels csv file
        with open(constants.building_values_file, "w") as f:
            f.write(f"parcelid,building_value,year,handwritten\n")
            for parcelid in downloaded_parcelids:
                f.write(f"'{parcelid},,,\n") # the ' preserves the value as string in google sheets

        # Postprocessing
        for file in os.listdir(constants.data_dir):
            if not file.endswith(".jpg"):
                continue

            if args.verbose:
                print(f"Postprocessing {file}")

            img_path = os.path.join(constants.data_dir, file)
            img = cv2.imread(img_path)

            # Cropping
            img = crop_image(img)

            # The image should be in landscape so if height > width, rotate the image based on orientation
            if img.shape[0] > img.shape[1]:
                print(f"Image {file} is in the wrong orientation")
                img = fix_orientation(img) # correct orientation if needed (some images are scanned in weird direction)

            cv2.imwrite(img_path, img)

    if not args.skip_upload:
        # Upload to GDrive
        current_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

        # ErukaLabels folder id
        labels_dir_id = "1WJ50iIVfCKRFPYOjBPvCYaSke8JOiLWz"

        # Create batch folder
        batch_folder = drive.CreateFile({
            "title": current_timestamp,
            "parents": [{"id": labels_dir_id}],
            "mimeType": "application/vnd.google-apps.folder"
        })
        batch_folder.Upload()

        # Upload csv
        print(f"Uploading {constants.building_values_name} to GDrive")
        csv_file = drive.CreateFile({
            "title": constants.building_values_name,
            "parents": [{"id": batch_folder.get('id')}],
            "mimeType": "text/csv"
        })
        csv_file.SetContentFile(constants.building_values_file)
        csv_file.Upload()

        for file in os.listdir(constants.data_dir):
            if file.endswith(".jpg"):
                print(f"Uploading {file} to GDrive")
                jpg_file = drive.CreateFile({
                    "title": file,
                    "parents": [{"id": batch_folder.get('id')}],
                    "mimeType": "image/jpeg"
                })
                jpg_file.SetContentFile(os.path.join(constants.data_dir, file))
                jpg_file.Upload()

        print(f"Label files ready on GDrive at ErukaLabels/{current_timestamp}")

