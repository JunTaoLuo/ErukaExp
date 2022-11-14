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
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import imutils
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.feature import canny
from skimage.draw import line
from operator import itemgetter

# Set PyTesseract Executable path (Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# TrOCR pre-trained model (imported using huggingface's transformers package)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten") 
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Hyperparameters for Tesseract
confidence_threshold = 60
width_tolerance = 0.3
min_entry_width = 10
max_entry_width = 250

# Parameters/filepaths
input_dir = "Dataset/Ownership/jpg"
output_dir = "Dataset/Ownership/ocr/"
log_dir = "Dataset/Ownership/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(f'{output_dir}/entry_boxes'):
    os.makedirs(f'{output_dir}/entry_boxes')

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

def binarize_image(img):
    thresh, thresh_image = cv2.threshold(img,127,255,cv2.THRESH_BINARY) # set a threshold based on which image converted to black/white
    thresh_image = cv2.convertScaleAbs(thresh_image) # converting the scale
    return thresh_image

def crop_building_column(img, building_indices, land_indices, total_indices, d):
    '''
    Inputs:
        - img: cropped ownership card image to remove big whitespace
        - building, land, and total indices: indices in the Pytesseract dict where these words were detected
        - d: dictionary result of the PyTesseract run, containing coordinates and output for each 'index'
    Outputs:
        - Cropped image of just the building column
    '''

    # Chosen index
    index = None
    
    # Coordinates to crop image
    width_param_left = 80 # add to the width on the left of where 'building' is detected
    width_param_right = 160 # add to width on right of where 'building' text is detected
    height_top_add = 15 # add to height of top of where 'building' is detected
    height_bottom_cut = 50 # cut irrelevant footer from the bottom of the image

    # If text 'building' not found, parameters that will be manipulated in if conditions below 
    # to further adjust from other words (total) to approximate position of 'building'
    horizontal_adjust = 0

    # If building not found but total was found, set a horizontal adjust to adjust the crop to building
    # column based on location of 'total'
    if len(total_indices) > 0 and len(building_indices) == 0:
        horizontal_adjust = -180 # negative because buildings is to the left of total
        index = total_indices[0] # which index in the dictionary to refer to

        x0 = d['left'][index] + horizontal_adjust - width_param_left
        x1 = x0 + d['width'][index] + width_param_right
        y0 = d['top'][index] - d['height'][index] - height_top_add
        y1 = img.shape[0] - height_bottom_cut

    elif len(building_indices) > 0:
        index = building_indices[0]
        horizontal_adjust = 0 # no adjust required

        x0 = d['left'][index] + horizontal_adjust - width_param_left
        x1 = x0 + d['width'][index] + width_param_right
        y0 = d['top'][index] - d['height'][index] - height_top_add
        y1 = img.shape[0] - height_bottom_cut        

    # Note: not doing the same for land, because the ownership card also has land written 
    # in the bottom right: "sketch of land". Hard to tell if it's the correct 'land' column
    # we're looking for.
    else:
        # If didn't detect building or total, apply a rough approximation to where buildings column will be
        x0 = 750
        x1 = 1030
        y0 = 540
        y1 = 850

    # Crop image based on parameters
    return img[y0:y1, x0:x1]

def hough_line_detection(img):
    '''
    Input: image of building column (cropped from whole card)
    Output: hspace, angles, dists of hough_line_peaks
    '''
    # Pre-processing for Hough
    y1_new = img.shape[0] - 750 # Further crop the image to the top part, because Hough works much better on smaller image
    img = img[:y1_new, :]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # grayscale
    thresh, thresh_image = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY) # binary thresholding
    gray = cv2.convertScaleAbs(thresh_image) # converting the scale
    edges = cv2.Canny(gray, 0, 200) # canny edge detection

    # Parameters for Hough
    thresh = 10
    min_distance = 20

    # Perform Hough line detection
    tested_angles = np.linspace(-np.pi, np.pi, 360, endpoint = False)
    h, theta, d = hough_line(edges, theta = tested_angles)
    hspace, angles, dists = hough_line_peaks(h, theta, d, thresh, min_distance)

    return hspace, angles, dists

def hough_inter(theta1, rho1, theta2, rho2):
    '''
    Calculate intersection point of a pair of lines in Polar coordinate space
    '''

    # Function sourced from this answer: https://stackoverflow.com/a/70371736
    A = np.array([[np.cos(theta1), np.sin(theta1)], 
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])
    
    return np.linalg.lstsq(A, b)[0] # use lstsq to solve Ax = b, not inv() which is unstable

def find_line_intersections(hspace, angles, dists, img):
    '''
    Input: result from hough line detection (hspace, angles, dists)
    Output: List of the relevant intersection points between lines
    '''
    # Repeating the cropping we did in earlier function, to make sure we're working with the same, smaller image
    y1_new = img.shape[0] - 750 
    img = img[:y1_new, :]

    # Segment found lines on image into vertical vs horizontal
    segmented = {}

    ## For each line found, classify into vertical or horizontal based on angle 
    for i in range(len(angles)):
        angle = angles[i]
        x = np.tan(angle + np.pi/2)
        if x >= 0:
            segmented[i] = 1 # horizontal line
        else:
            segmented[i] = 0 # vertical line

    # Now loop through all combinations of lines, only checking for intersections if they are of different types
    intersections = []
    for i in range(len(angles) - 1):
        for j in range(1, len(angles)):
            if segmented[i] != segmented[j]: # check that we are only running algo for vertical / horizontal line pairs
                point = hough_inter(angles[i], dists[i], angles[j], dists[j])
                intersections.append(point)
            else:
                pass
    
    # Subset intersection points to only the relevant ones that fall within the boundaries of the image
    relevant_intersects = [[x[0], x[1]] for x in intersections if (x[0] >= 0 and x[0] <= img.shape[0]) and (x[1] >= 0 and x[1] <= img.shape[1])]

    return relevant_intersects

def get_box_params(intersections, img):
    '''
    Input: 
        - list of relevant intersection points outputted by find_line_intersections
        - image of the building column
    Output:
        - Parameters:
            - x, y: coordinates of topleft corner
            - w, h, s: width, height, and slant of the column
    '''
    # Sort intersections from top to bottom on the page
    rinter = intersections
    rinter.sort(key = lambda x: x[1]) 

    # Get differences between each intersection and the one before
    x_ax_diffs = np.array([round(rinter[i+1][0] - rinter[i][0], 3) for i in range(len(rinter)-1)])
    y_ax_diffs = np.array([round(rinter[i+1][1] - rinter[i][1], 3) for i in range(len(rinter)-1)])

    ###### Width ########

    # Get width of box by looking at x_ax_diffs -> something within range of 150 - 250
    lower_bound_width = 150
    upper_bound_width = 200

    width_candidates = x_ax_diffs[(abs(x_ax_diffs) >= lower_bound_width) & (abs(x_ax_diffs) <= upper_bound_width)]

    # Select the one with the max value
    w_final = np.max(abs(width_candidates))

    # Check that this value is consistent; i.e., at least 4 different widths are within some error range of this
    tolerance = 5 # count width values that fall within 5 pixels of w_final
    close_count_w = len(width_candidates[np.where(abs(abs(width_candidates) - w_final) < tolerance)])

    if close_count_w < 4:
        # If it isn't consistent, just pick a default value that's a rough estimate
        w_final = 180
    # Else, just stick with the w_final we got using this method

    ###### Height ########

    # Get height of box by looking at diff in y_axis between 35-70
    lower_bound_height = 35
    upper_bound_height = 70

    tolerance = 5

    height_candidates = y_ax_diffs[(abs(y_ax_diffs) >= lower_bound_height) & (abs(y_ax_diffs) <= upper_bound_height)]

    h_final = np.max(abs(height_candidates))

    close_count_h = len(height_candidates[np.where(abs(abs(height_candidates) - h_final) < tolerance)])

    if close_count_h < 4:
        # Similar methodology as above: if we can't find a consistent answer, then use a rough approx
        h_final = 55

    ###### Slant ########
    y_ax_slant = np.round(y_ax_diffs, 1)

    slant_candidates = y_ax_slant[(abs(y_ax_slant) < 5) & (y_ax_slant != 0)]

    slant_final = np.max(slant_candidates) # not absolute value because slant can either be positive or negative

    close_count_s = len(slant_candidates[np.where(abs(slant_candidates) - abs(slant_final) < tolerance)])

    if close_count_s < 3:
        slant_final = 3

    ###### Top-left corner pixel coordinates ########
    rinter.sort(key = lambda x: (x[1], x[0]))
    topleft = rinter[0]

    # If the coordinate isn't within reasonable values, then assign it roughly
    if topleft[0] < 45 or topleft[0] > 60 or topleft[1] > 10:
        topleft = [53, 5]
    

    return topleft[0], topleft[1], w_final, h_final, slant_final

def mark_intersects(img, x, y, w, h, s):
    '''
    Inputs: image of building column. (x,y) of topleft corner. width, height and slant of column.
    Outputs: image marking the rectangles used
    '''
    # Each column has 20 boxes
    for i in range(20):
        img = cv2.rectangle(img, (x, y), (x+w, y+h+s), (0, 255, 0), 1)
        y = y + h + s

    return img

def filter_entry_texts_core(text_raw):
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
    filtered_text = ""
    for char in text_raw:
        if char in match_list:
            filtered_text += char
        if char in substitutions:
            filtered_text += substitutions[char]
    return filtered_text

def filter_entry_texts(entries_list2D, entries_conf_list2D):
    all_texts = []
    all_conf = []
    for entriesIndex, entry_list in enumerate(entries_list2D):
        entry_texts = []
        entry_conf = []
        for index, text_raw in enumerate(entry_list):
            if text_raw:
                filtered_text = filter_entry_texts_core(text_raw)
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
        self.initial_building_value_trocr= 0
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

    # Run tesseract to get locations of building/total/land (indices in the dict)
    d = pytesseract.image_to_data(img, output_type=Output.DICT)

    result = ParcelResult(parcel)

    for index, text in enumerate(d['text']):
        if text.lower() == "buildings":
            result.building_indicies.append(index)
        if text.lower() == "total":
            result.total_indicies.append(index)
        if text.lower() == "land":
            result.land_indicies.append(index)

    # Crop just the building column
    img = crop_building_column(img, result.building_indicies, result.land_indicies, result.total_indicies, d)

    # Run Hough line detection
    hspace, angles, dists = hough_line_detection(img)

    # Find relevant intersection points
    intersects = find_line_intersections(hspace, angles, dists, img)

    # Get parameters of the boxes in the building column
    x, y, w, h, s = get_box_params(intersects, img)

    # Mark the boxes formed by the intersections, and write image to disk
    marked_img = mark_intersects(img, x, y, w, h, s)

    # Write marked image to file for diagnostics
    cv2.imwrite(f'{output_dir}/{parcel}.jpg')

    # Run TR-OCR on the sub-images
    building_texts_trocr_raw=[]
    building_texts_trocr=[]

    trocr_text_raw = []
    trocr_text = []

    # There are 20 boxes in the building column
    for i in range(20): 
        # Subset to one box
        entry_img = img[y:y+h+s, x:x+w]
        cv2.imwrite(f'{output_dir}/entry_boxes/{parcel}-{i}.jpg')

        # Try TR-OCR
        print(f"Trying TrOCR on {output_dir}/entry_boxes/{parcel}-{i}.jpg")
        pixel_values = processor(entry_img, return_tensors="pt").pixel_values 
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        filtered_text = filter_entry_texts_core(generated_text)
        if generated_text:
            trocr_text_raw.append(generated_text)
            if filtered_text:
                parsed_value = int(filtered_text)
                trocr_text.append(filtered_text)
                result.initial_building_value_trocr = parsed_value
            else:
                trocr_text.append("")

        
        # Move onto next box
        y = y + h + s

    building_texts_trocr_raw.append(trocr_text_raw)  
    building_texts_trocr.append(trocr_text)  

    # Write information to log
    with open(f'{output_dir}/{parcel}.log', "w") as f:
        f.write(f"Found buildings: {len(result.building_indicies)}\n")

        f.write(f"Found total: {len(result.total_indicies)}\n")

        if len(building_texts_trocr) > 0:
            f.write(f"TrOCR text raw:\n")
            for building_index, building_texts in enumerate(building_texts_trocr_raw):
                f.write(f"  Building column {building_index}:\n")
                for text_index, text in enumerate(building_texts):
                    f.write(f"    {text}\n")
            f.write(f"TrOCR text:\n")
            for building_index, building_texts in enumerate(building_texts_trocr):
                f.write(f"  Building column {building_index}:\n")
                for text_index, text in enumerate(building_texts):
                    f.write(f"    {text}\n")

    return result

### Main operations in the file

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
building_value_parsed = sum(1 if r.initial_building_value > 0 or r.initial_building_value_trocr > 0 else 0 for r in results)
building_accurate = 0

correct_results = ""
incorrect_results = ""

for r in results:
    if r.parcel in targets:
        target_value = targets[r.parcel].initial_building_value
        parsed_value = r.initial_building_value
        resolved_value = parsed_value if parsed_value >= 100 and parsed_value % 10 == 0 else r.initial_building_value_trocr
        if isclose(target_value, resolved_value, rel_tol=0.2):
            building_accurate += 1
            correct_results += f"Accurate OCR result for parcel: {r.parcel}, target: {target_value} result: {resolved_value} tesseract: {parsed_value} confidence: {r.initial_building_value_confidence} trocr: {r.initial_building_value_trocr}\n"
        else:
            incorrect_results += f"Inaccurate OCR result for parcel: {r.parcel}, target: {target_value} result: {resolved_value} tesseract: {parsed_value} confidence: {r.initial_building_value_confidence} trocr: {r.initial_building_value_trocr}\n"

print(correct_results)
print(incorrect_results)

print(f"Statistics:")
print(f"Total parcels processed: {len(results)}")
print(f"Recognized land: {land_recognized}, building: {building_recognized}, total: {total_recognized}, any: {any_recognized}")
print(f"Inferred building: {building_inferred}")
print(f"Parsed building: {building_value_parsed}")
print(f"Errors multiple land: {multiple_land_recognized}, multiple building: {multiple_building_recognized}, multiple total: {multiple_total_recognized}")
print(f"Accurate building: {building_accurate}")