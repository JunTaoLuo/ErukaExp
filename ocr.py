import time
import cv2
import os
import pytesseract
import numpy as np
from pytesseract import Output
from pathlib import Path

# Set PyTesseract Executable path
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

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
    match_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ","]
    all_texts = []
    for entry_list in entries_list2D:
        entry_texts = []
        for text_raw in entry_list:
            matched = [c in match_list for c in text_raw]
            if all(matched):
                text_raw.replace(",", "")
                text_raw.replace(".", "")
                entry_texts.append(text_raw)
        all_texts.append(entry_texts)
    return all_texts

def parse_parcel(parcel):

    img = cv2.imread(f'{input_dir}/{parcel}.jpg')

    # Image pre-processing
    img = crop_image(img) # cropping before running pytesseract improves the speed dramatically
    img = binarize_image(img) 

#    img = remove_noise(img) # Note: noise removal doesn't really work well, it makes the image blurrier. Also because there's not much noise to remove.
#    img = cv2.Canny(img,0,200) # Edge detection also doesn't really help
    
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

    cv2.imwrite(f'{output_dir}/{parcel}.jpg', img)

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
