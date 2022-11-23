import os
import shutil

# Parameters
input_dir = "Dataset/Ownership/ocr"
output_dir = "Dataset/Ownership/ocr-test"

for file in os.listdir(input_dir):
    if file.endswith("-0.jpg"):
        parcel = file[:-6]
        print(f"Copying file {parcel}")
        shutil.copy(f"{input_dir}/{file}", f"{output_dir}/{parcel}.jpg")