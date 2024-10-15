import cv2
import numpy as np
from rembg import remove
from PIL import Image
import os
import pickle

def remove_background(input_image_path):
    input_image = Image.open(input_image_path)
    output_image = remove(np.array(input_image))
    output_image = Image.fromarray(output_image)
    output_image_cv = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGBA2BGR)
    return output_image_cv

def extract_mask(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    return mask

def detect_and_save_features(mask, feature_type, directory, index):
    if feature_type == 'SIFT':
        feature_detector = cv2.SIFT_create()
    elif feature_type == 'SURF':
        feature_detector = cv2.xfeatures2d.SURF_create()
    elif feature_type == 'ORB':
        feature_detector = cv2.ORB_create()
    else:
        raise ValueError("Unsupported feature type")

    keypoints, descriptors = feature_detector.detectAndCompute(mask, None)
    filename = f"contour_{feature_type}_{index}.pkl"

    # Save descriptors to a file
    with open(os.path.join(directory, filename), 'wb') as f:
        pickle.dump((keypoints, descriptors), f)

def setup_directories():
    directory = 'contour_saved'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def main():
    input_image_path = 'path_to_your_image.jpg'
    background_removed_image = remove_background(input_image_path)
    mask = extract_mask(background_removed_image)
    directory = setup_directories()

    # You might want to manage index externally or check existing files to determine the next index
    for feature_type in ['SIFT', 'SURF', 'ORB']:
        # Assume you maintain an index for each feature type
        index = 0  # This should be dynamically managed based on existing files
        detect_and_save_features(mask, feature_type, directory, index)

if __name__ == "__main__":
    main()
