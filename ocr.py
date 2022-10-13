import time
import cv2
import os
import pytesseract
from pytesseract import Output
from pathlib import Path


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
def get_entries(column_header_indicies, d):
    all_entries = []
    for columns_i in column_header_indicies:
        column_entries = []
        # print(f"column left: {d['left'][columns_i]}, right: {d['left'][columns_i] + d['width'][columns_i]}, top: {d['top'][columns_i]}")
        for i in range(len(d['text'])):
            if d['left'][columns_i] - d['width'][columns_i]*width_tolerance < d['left'][i] and d['left'][columns_i] + d['width'][columns_i] > d['left'][i] and d['top'][columns_i] < d['top'][i]:
                if min_entry_width < d['width'][i] and d['width'][i] < max_entry_width:
                    # print(f"entry left: {d['left'][i]} top: {d['top'][i]} width: {d['width'][i]}")
                    column_entries.append(i)

        column_entries.sort(key=lambda b: d['top'][b])
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
    for entry_list in entries_list2D:
        entry_text = []
        for i in entry_list:
            entry_text.append(d['text'][i])
        all_texts.append(entry_text)
    return all_texts

def filter_entry_texts(entries_list2D):
    match_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]
    all_texts = []
    for entry_list in entries_list2D:
        entry_texts = []
        for text_raw in entry_list:
            matched = [c in match_list for c in text_raw]
            if all(matched):
                entry_texts.append(text_raw.replace(".", ""))
        all_texts.append(entry_texts)
    return all_texts

def parse_parcel(parcel):

    img = cv2.imread(f'{input_dir}/{parcel}.jpg')

    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    # print(d.keys())
    # print(d['text'])

    buildings_indicies = []
    total_indicies = []

    for index, text in enumerate(d['text']):
        if text.lower() == "buildings":
            buildings_indicies.append(index)
        if text.lower() == "total":
            total_indicies.append(index)

    # img = mark_indicies_list(range(len(d['text'])), img, d)
    img = mark_indicies_list(buildings_indicies, img, d)
    img = mark_indicies_list(total_indicies, img, d)

    buildings_entries = get_entries(buildings_indicies, d)
    totals_entries = get_entries(total_indicies, d)

    img = mark_indicies_list2D(buildings_entries, img, d)
    img = mark_indicies_list2D(totals_entries, img, d)

    img_dim = img.shape
    cropped_img = img[int(img_dim[0]*0.75):img_dim[0], 0:int(img_dim[1]*0.25)]
    cv2.imwrite(f'{output_dir}/{parcel}.jpg', cropped_img)

    buildings_texts_raw = get_entry_texts(buildings_entries, d)
    totals_texts_raw = get_entry_texts(totals_entries, d)

    buildings_texts = filter_entry_texts(buildings_texts_raw)
    totals_texts = filter_entry_texts(totals_texts_raw)

    parsed_building = ""
    parsed_total = ""

    with open(f'{output_dir}/{parcel}.log', "w") as f:
        f.write(f"Found buildings: {len(buildings_indicies)}\n")
        f.write(f"Found total: {len(total_indicies)}\n")

        f.write(f"Raw text:\n")
        for building_index, building_texts in enumerate(buildings_texts_raw):
            f.write(f"  Building column {building_index}:\n")
            for text in building_texts:
                f.write(f"    {text}\n")
        for total_index, total_texts in enumerate(totals_texts_raw):
            f.write(f"  Total column {total_index}:\n")
            for text in total_texts:
                f.write(f"    {text}\n")

        f.write(f"Filtered text:\n")
        for building_index, building_texts in enumerate(buildings_texts):
            f.write(f"  Building column {building_index}:\n")
            for text in building_texts:
                f.write(f"    {text}\n")
        for total_index, total_texts in enumerate(totals_texts):
            f.write(f"  Total column {total_index}:\n")
            for text in total_texts:
                f.write(f"    {text}\n")

        f.write(f"Parse result:\n")
        for building_index, building_texts in enumerate(buildings_texts):
            if len(building_texts) > 0:
                f.write(f"  Building column {building_index}: {building_texts[0]}\n")
                parsed_building = building_texts[0]
        for total_index, total_texts in enumerate(totals_texts):
            if len(total_texts) > 0:
                f.write(f"  Total column {total_index}: {total_texts[0]}\n")
                parsed_total = total_texts[0]

    return (parsed_building, parsed_total)

parcels = []

for file in os.listdir(input_dir):
    parcel = file[:-4]
    parcels.append(parcel)

parcels.sort()
# parcels = ["0180001007300"]

results = []
for parcel in parcels:
    start = time.time()
    building, total = parse_parcel(parcel)
    end = time.time()
    print(f"Parcel {parcel} Building: {building} Total: {total} - {end-start:.3f}s")
    results.append((parcel, building, total))

    # Only process 80 entries for now
    if len(results) == 80:
        break

with open(f"{log_dir}/ocr.log", "w") as f:
    for (parcel, building, total) in results:
        f.write(f"{parcel},{building},{total}\n")
