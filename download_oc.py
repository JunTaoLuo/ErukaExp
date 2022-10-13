import os
import pandas as pd
import pdf2image as p2i
import re
import requests as rq

# input: Dataset/oc_input.csv - a list of parcel numbers
input = pd.read_csv("Dataset/oc_input.csv").to_numpy()
results_output_file = "Dataset/Ownership/results.csv"
output_dir = "Dataset/Ownership/jpg/"

# Ownership page source template
ownership_source = 'https://wedge.hcauditor.org/view/re/{}/2021/imagesOwnerCard'
ownership_reference_regex = '.*convert\/(.*.tif)\/.*'
ownership_file = 'https://wedge.hcauditor.org/convert/{}/'


def download_file(parcel_number):
    print(f"Processing {parcel_number}")

    # Retrieve ownership card page source (HTML)
    ownership_source_formatted = ownership_source.format(parcel_number)
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
        print(f"Failed to find ownership card reference in {response.text}")
        return "PDF reference not found"

    # Download PDF
    ownership_file_formated = ownership_file.format(oc_reference)
    print(f"Downloading ownership card PDF {ownership_file_formated}")

    response = rq.get(ownership_file_formated)
    if not response.ok:
        print(f"Failed to get {ownership_file_formated}")
        return "PDF download failure"

    # with open(f"Dataset/Ownership/pdf/{parcel_number}.pdf", "wb") as f:
    #     f.write(response.content)

    # Convert to jpg
    pages = p2i.convert_from_bytes(response.content)

    if len(pages) == 0:
        print("Failed to convert PDF to JPEG")
        return "JPEG conversion failure"

    if len(pages) == 1:
        jpeg_file = f"Dataset/Ownership/jpg/{parcel_number}.jpg"
        if os.path.exists(jpeg_file):
            os.remove(jpeg_file)
        pages[0].save(jpeg_file, "JPEG")
    else:
        for i, page in enumerate(pages):
            jpeg_file = f"Dataset/Ownership/jpg/{parcel_number}-{i}.jpg"
            if os.path.exists(jpeg_file):
                os.remove(jpeg_file)
            page.save(jpeg_file, "JPEG")

    return "Success"

# Results
file_results = []
summary_results = {}

for entry in input:
    result = download_file(entry[0])
    print(result)

    file_results.append((entry[0], result))
    if result in summary_results:
        summary_results[result] = summary_results[result] + 1
    else:
        summary_results[result] = 1

print(summary_results)

with open(results_output_file, "w") as f:
    for (entry, result) in file_results:
        f.write(f"{str(entry)},{str(result)}\n")