import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import stag
import pickle

class ExtractFeatures:
    def __init__(self, image_path, stag_id):
        self.image_path = image_path
        self.stag_id = stag_id
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError("Image could not be loaded.")
        self.corners = None
        self.ids = None
        self.homogenized_image = None

    def detect_stag(self):
        config = {'libraryHD': 17, 'errorCorrection': -1}
        self.corners, self.ids, _ = stag.detectMarkers(self.image, **config)
        if self.ids is not None and self.stag_id in self.ids:
            index = np.where(self.ids == self.stag_id)[0][0]
            self.corners = self.corners[index].reshape(-1, 2)
            return True
        print("Marker with ID", self.stag_id, "not found.")
        return False

    def homogenize_image_based_on_corners(self):
        if self.corners is None:
            return None
        x, y, w, h = cv2.boundingRect(self.corners.astype(np.float32))
        aligned_corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype='float32')
        transform_matrix = cv2.getPerspectiveTransform(self.corners, aligned_corners)
        self.homogenized_image = cv2.warpPerspective(self.image, transform_matrix, (self.image.shape[1], self.image.shape[0]))
        return self.homogenized_image

    def create_mask(self):
        if self.homogenized_image is None:
            return None
        gray_image = cv2.cvtColor(self.homogenized_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        return mask

    def find_and_draw_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(self.homogenized_image, [largest_contour], -1, (255, 0, 0), 2)
            return largest_contour
        return None

    def compute_chain_code(self, contour):
        chain_code = []
        for i in range(1, len(contour)):
            p1 = tuple(contour[i - 1][0])
            p2 = tuple(contour[i][0])
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            direction = (np.sign(dx), np.sign(dy))
            chain_code.append(direction)
        return chain_code

class LoadContours:
    def __init__(self, contour_path, chain_code_path):
        self.contour_path = contour_path
        self.chain_code_path = chain_code_path

    def load_data(self):
        with open(self.contour_path, 'rb') as f:
            contours = pickle.load(f)
        with open(self.chain_code_path, 'rb') as f:
            chain_code = pickle.load(f)
        return contours, chain_code

def compare_features(extracted_contour, loaded_contour, extracted_chain_code, loaded_chain_code):
    # Compare contours
    iou_contour = calculate_contour_iou(extracted_contour, loaded_contour)
    print(f"Contour IoU: {iou_contour}")

    # Compare chain codes
    iou_chain_code = calculate_chain_code_iou(extracted_chain_code, loaded_chain_code)
    print(f"Chain Code IoU: {iou_chain_code}")

def calculate_contour_iou(contour1, contour2):
    # Simplistic IoU calculation (requires actual implementation based on contour comparison)
    return np.random.random()  # Placeholder for demonstration

def calculate_chain_code_iou(chain_code1, chain_code2):
    # Simplistic IoU calculation (requires actual implementation based on chain code comparison)
    return np.random.random()  # Placeholder for demonstration

if __name__ == "__main__":
    image_path = "./frames/thiago_fotos_10_feature_afternoon/img_0_009.jpg"
    stag_id = 0
    processor = ExtractFeatures(image_path, stag_id)
    if processor.detect_stag():
        homogenized = processor.homogenize_image_based_on_corners()
        if homogenized is not None:
            plt.imshow(cv2.cvtColor(homogenized, cv2.COLOR_BGR2RGB))
            plt.title('Homogenized Image')
            plt.show()

            mask = processor.create_mask()
            if mask is not None:
                largest_contour = processor.find_and_draw_contours(mask)
                if largest_contour is not None:
                    plt.imshow(cv2.cvtColor(homogenized, cv2.COLOR_BGR2RGB))
                    plt.title('Processed Image with Contour')
                    plt.show()

                    chain_code = processor.compute_chain_code(largest_contour)

                    loader = LoadContours('features/contours/contour_0.pkl', 'features/chain_code/chain_code_1.pkl')
                    loaded_contour, loaded_chain_code = loader.load_data()
                    compare_features(largest_contour, loaded_contour, chain_code, loaded_chain_code)
