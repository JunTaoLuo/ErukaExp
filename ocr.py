from math import isclose
import time
import csv
import cv2
import os
import pytesseract
import numpy as np
from pytesseract import Output
from pathlib import Path
from tqdm import tqdm
import imutils

# Set PyTesseract Executable path (Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Hyperparameters
confidence_threshold = 60
width_tolerance = 0.3
min_entry_width = 10
max_entry_width = 250

# Parameters
input_dir = "Dataset/Ownership/jpg"
output_dir = "Dataset/Ownership/ocr/"
log_dir = "Dataset/Ownership/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Utilties
def crop_image(img):
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
    results = pytesseract.image_to_osd(img, output_type=Output.DICT)

    if results['orientation'] == 0: # if image is already in correct orientation, do nothing
        return img
    else:
        return imutils.rotate_bound(img, angle=results["rotate"]) # rotate by the amount identified by pytesseract

def remove_noise(img):
    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    # Apply blur to smooth out the edges
    img = cv2.GaussianBlur(img, (5, 5), 0)

    return img

def binarize_image(img):
    thresh, thresh_image = cv2.threshold(img,127,255,cv2.THRESH_BINARY) # set a threshold based on which image converted to black/white
    thresh_image = cv2.convertScaleAbs(thresh_image) # converting the scale
    return thresh_image

def get_entries_core(building_left, building_top, building_width, d):
    column_entries = []
    # print(f"column left: {building_left}, right: {building_left + building_width}, top: {building_top}")
    for i in range(len(d['text'])):
        if building_left - building_width*width_tolerance < d['left'][i] and building_left + building_width > d['left'][i] and building_top < d['top'][i]:
            if min_entry_width < d['width'][i] and d['width'][i] < max_entry_width:
                # print(f"entry left: {d['left'][i]} top: {d['top'][i]} width: {d['width'][i]}")
                column_entries.append(i)

    column_entries.sort(key=lambda b: d['top'][b])
    return column_entries

def get_entries(column_header_indicies, d):
    all_entries = []
    for columns_i in column_header_indicies:
        column_entries = get_entries_core(d['left'][columns_i], d['top'][columns_i], d['width'][columns_i], d)
        all_entries.append(column_entries)
    return all_entries

def mark_indicies_list(indicies_list, img, d):
    for i in indicies_list:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return img

def mark_indicies_list2D(indicies_list2D, img, d):
    for indicies_list in indicies_list2D:
        for i in indicies_list:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return img

def get_entry_texts(entries_list2D, d):
    all_texts = []
    all_confidence = []
    for entry_list in entries_list2D:
        i = 0
        entry_text = []
        entry_confidence = []
        while i < len(entry_list):
            # From inspection, each row has a height of about 50
            # print(f"text: {d['text'][i]} left: {d['left'][i]}, right: {d['left'][i] + d['width'][i]}, top: {d['top'][i]}")

            row_top = d['top'][entry_list[i]]
            row_entries = [entry_list[i]]
            i += 1

            # Append indicies that are in the same row
            while i < len(entry_list):
                if d['top'][entry_list[i]] < row_top + 10:
                    row_entries.append(entry_list[i])
                else:
                    break
                i += 1

            # Sort in left to right order
            row_entries.sort(key=lambda x: d['left'][x])

            # Construct row text
            row_text = ""
            row_confidence = 0.0
            for element_index in row_entries:
                row_text += d['text'][element_index]
                row_confidence += d['conf'][element_index]
            confidence = row_confidence / len(row_entries)

            # Add row text
            entry_text.append(row_text)
            entry_confidence.append(confidence)

        all_texts.append(entry_text)
        all_confidence.append(entry_confidence)
    return all_texts, all_confidence

def filter_entry_texts(entries_list2D, entries_conf_list2D):
    # TODO: Add more entries here
    substitutions = {
        "G": "6",
        "$": "5",
        "¥": "4",
        "(": "1",
        "/": "1",
        "I": "1",
        "o": "0",
        "O": "0",
        "D": "0",
    }
    match_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    all_texts = []
    all_conf = []
    for entriesIndex, entry_list in enumerate(entries_list2D):
        entry_texts = []
        entry_conf = []
        for index, text_raw in enumerate(entry_list):
            if text_raw:
                filtered_text = ""
                for char in text_raw:
                    if char in match_list:
                        filtered_text += char
                    if char in substitutions:
                        filtered_text += substitutions[char]
                if filtered_text:
                    entry_texts.append(filtered_text)
                    entry_conf.append(entries_conf_list2D[entriesIndex][index])
        all_texts.append(entry_texts)
        all_conf.append(entry_conf)
    return all_texts, all_conf

class ParcelResult():
    def __init__(self, parcel) -> None:
        self.parcel = parcel
        self.building_indicies = []
        self.land_indicies = []
        self.total_indicies = []
        self.inferred_building = False
        self.inferred_building_coordinates = None
        self.initial_building_value = 0
        self.initial_building_value_confidence = 0.0
        self.initial_land_value = 0

class TargetLabel():
    def __init__(self, parcel) -> None:
        self.parcel = parcel
        self.initial_building_value = 0
        self.initial_land_value = 0

def parse_parcel(parcel) -> ParcelResult:

    img = cv2.imread(f'{input_dir}/{parcel}.jpg')

    # Image pre-processing
    img = crop_image(img) # cropping before running pytesseract improves the speed dramatically

    # The image should be in landscape so if height > width, rotate the image based on orientation
    if img.shape[0] > img.shape[1]:
        print(f"Parcel {parcel} is in the wrong orientation")
        img = fix_orientation(img) # correct orientation if needed (some images are scanned in weird direction)

    img = binarize_image(img)

    # img = remove_noise(img) # Note: noise removal doesn't really work well, it makes the image blurrier. Also because there's not much noise to remove.
    # img = cv2.Canny(img,0,200) # Edge detection also doesn't really help

    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    # d = pytesseract.image_to_data(img, output_type=Output.DICT, config="-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    # print(d.keys())
    # print(d['text'])

    result = ParcelResult(parcel)

    for index, text in enumerate(d['text']):
        if text.lower() == "buildings":
            result.building_indicies.append(index)
        if text.lower() == "total":
            result.total_indicies.append(index)
        if text.lower() == "land":
            result.land_indicies.append(index)

    # img = mark_indicies_list(range(len(d['text'])), img, d)
    img = mark_indicies_list(result.building_indicies, img, d)
    img = mark_indicies_list(result.land_indicies, img, d)
    img = mark_indicies_list(result.total_indicies, img, d)

    buildings_entries = get_entries(result.building_indicies, d)
    # lands_entries = get_entries(result.land_indicies, d)

    if len(result.building_indicies) == 0 and len(result.total_indicies) > 0:
        result.inferred_building = True
        # From inspection left is about 200 less than total's left, tops are the same, and width is 127
        inferred_left = d['left'][result.total_indicies[0]] - 200
        inferred_top = d['top'][result.total_indicies[0]]
        inferred_width = 127
        result.inferred_building_coordinates = (inferred_left, inferred_top, inferred_width)
        buildings_entries = [get_entries_core(inferred_left, inferred_top, inferred_width, d)]

    img = mark_indicies_list2D(buildings_entries, img, d)
    # img = mark_indicies_list2D(lands_entries, img, d)

    cv2.imwrite(f'{output_dir}/{parcel}.jpg', img)

    buildings_texts_raw, buildings_texts_raw_conf = get_entry_texts(buildings_entries, d)

    # lands_texts_raw = get_entry_texts(lands_entries, d)

    buildings_texts, buildings_texts_conf = filter_entry_texts(buildings_texts_raw, buildings_texts_raw_conf)
    # lands_texts = filter_entry_texts(lands_texts_raw)

    with open(f'{output_dir}/{parcel}.log', "w") as f:
        f.write(f"Found buildings: {len(result.building_indicies)}\n")
    #     f.write(f"Found land: {len(result.land_indicies)}\n")

        if result.inferred_building:
            left, top, width = result.inferred_building_coordinates
            f.write(f"Inferred building coordinates left: {left} top: {top} width: {width}\n")

        f.write(f"Raw text:\n")
        for building_index, building_texts in enumerate(buildings_texts_raw):
            f.write(f"  Building column {building_index}:\n")
            for text_index, text in enumerate(building_texts):
                f.write(f"    {text}({buildings_texts_raw_conf[building_index][text_index]})\n")
    #     for land_index, land_texts in enumerate(lands_texts_raw):
    #         f.write(f"  Land column {land_index}:\n")
    #         for text in land_texts:
    #             f.write(f"    {text}\n")

        f.write(f"Filtered text:\n")
        for building_index, building_texts in enumerate(buildings_texts):
            f.write(f"  Building column {building_index}:\n")
            for text_index, text in enumerate(building_texts):
                f.write(f"    {text}({buildings_texts_conf[building_index][text_index]})\n")
    #     for land_index, land_texts in enumerate(lands_texts):
    #         f.write(f"  Land column {land_index}:\n")
    #         for text in land_texts:
    #             f.write(f"    {text}\n")

        f.write(f"Parse result:\n")
        for building_index, building_texts in enumerate(buildings_texts):
            if len(building_texts) > 0:
                f.write(f"  Building column {building_index}: {building_texts[0]}\n")
                result.initial_building_value = int(building_texts[0])
                result.initial_building_value_confidence = buildings_texts_conf[building_index][0]
    #     for land_index, land_texts in enumerate(lands_texts):
    #         if len(land_texts) > 0:
    #             f.write(f"  Land column {land_index}: {land_texts[0]}\n")
    #             result.initial_land_value = land_texts[0]

    # if len(result.building_indicies) > 0:
    #     print(f"building left: {d['left'][result.building_indicies[0]]} top: {d['top'][result.building_indicies[0]]} width: {d['width'][result.building_indicies[0]]}")
    # if len(result.total_indicies) > 0:
    #     print(f"total left: {d['left'][result.total_indicies[0]]} top: {d['top'][result.total_indicies[0]]} width: {d['width'][result.total_indicies[0]]}")

    return result

parcels = []

for file in os.listdir(input_dir):
    parcel = file[:-4]
    parcels.append(parcel)

parcels.sort()
# parcels = ["0390003004600"]

targets: dict[str, TargetLabel] = {}

with open("Dataset/Ownership/buildings.csv", "r") as f:
    building_labels = csv.DictReader(f)

    for row in building_labels:
        parcel = row["parcel"]
        building_value = row["value"]

        if building_value:
            label = TargetLabel(parcel)
            label.initial_building_value = int(building_value)
            targets[parcel] = label

with open("Dataset/Ownership/land.csv", "r") as f:
    land_labels = csv.DictReader(f)

    for row in land_labels:
        parcel = row["parcel"]
        land_value = row["value"]

        if land_value:
            if parcel in targets:
                targets[parcel].initial_land_value = int(land_value)
            else:
                label = TargetLabel(parcel)
                label.initial_land_value = int(land_value)
                targets[parcel] = label

# for target in targets.values():
#     print(f"Parcel: {target.parcel}, Land: {target.initial_land_value}, Building: {target.initial_building_value}")

results: list[ParcelResult] = []

for i in tqdm(range(80)):
    parcel = parcels[i]
    start = time.time()
    result = parse_parcel(parcel)
    end = time.time()
    print(f"Parcel {parcel} Building: {result.initial_building_value} Land: {result.initial_land_value} - {end-start:.3f}s")
    results.append(result)

# with open(f"{log_dir}/ocr.log", "w") as f:
#     for (parcel, building, total) in results:
#         f.write(f"{parcel},{building},{total}\n")

land_recognized = sum(1 if len(r.land_indicies) > 0 else 0 for r in results)
building_recognized = sum(1 if len(r.building_indicies) > 0 else 0 for r in results)
total_recognized = sum(1 if len(r.total_indicies) > 0 else 0 for r in results)
any_recognized = sum(1 if (len(r.building_indicies) > 0 or len(r.total_indicies) > 0) else 0 for r in results)
building_inferred = sum(1 if r.inferred_building > 0 else 0 for r in results)
multiple_land_recognized = sum(1 if len(r.land_indicies) > 1 else 0 for r in results)
multiple_building_recognized = sum(1 if len(r.building_indicies) > 1 else 0 for r in results)
multiple_total_recognized = sum(1 if len(r.total_indicies) > 1 else 0 for r in results)
building_value_parsed = sum(1 if r.initial_building_value > 0 else 0 for r in results)
building_accurate = 0

correct_results = ""
incorrect_results = ""

for r in results:
    if r.parcel in targets:
        target_value = targets[r.parcel].initial_building_value
        parsed_value = r.initial_building_value
        if isclose(target_value, parsed_value, rel_tol=0.2):
            building_accurate += 1
            correct_results += f"Accurate OCR result for parcel: {r.parcel}, target: {target_value} result: {parsed_value} confidence: {r.initial_building_value_confidence}\n"
        else:
            incorrect_results += f"Inaccurate OCR result for parcel: {r.parcel}, target: {target_value} result: {parsed_value} confidence: {r.initial_building_value_confidence}\n"

print(correct_results)
print(incorrect_results)

print(f"Statistics:")
print(f"Total parcels processed: {len(results)}")
print(f"Recognized land: {land_recognized}, building: {building_recognized}, total: {total_recognized}, any: {any_recognized}")
print(f"Inferred building: {building_inferred}")
print(f"Parsed building: {building_value_parsed}")
print(f"Errors multiple land: {multiple_land_recognized}, multiple building: {multiple_building_recognized}, multiple total: {multiple_total_recognized}")
print(f"Accurate building: {building_accurate}")
