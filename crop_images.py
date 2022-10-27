import cv2
import os
import numpy as np

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

if __name__ == '__main__':

    # Parametrize these later 
    input_dir = "Dataset/Ownership/jpg"
    output_dir = "Dataset/Ownership/cropped"

    for file in os.listdir(input_dir):
        img = cv2.imread(f'{input_dir}/{file}')
        img = crop_image(img)
        cv2.imwrite(f'{output_dir}/{file}', img)

