import cv2
import numpy as np
import matplotlib.pyplot as plt
import stag
from rembg import remove
import os

class ExtractFeatures:
    def __init__(self, image_path, stag_id):
        self.image_path = image_path
        self.stag_id = stag_id
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError("Error loading image.")
        self.corners = None
        self.ids = None
        self.output_directory = "./output"
        self.file_counter = 0

    def detect_stag(self):
        config = {'libraryHD': 17, 'errorCorrection': -1}
        self.corners, self.ids, _ = stag.detectMarkers(self.image, **config)
        if self.ids is not None and self.stag_id in self.ids:
            index = np.where(self.ids == self.stag_id)[0][0]
            self.corners = self.corners[index].reshape(-1, 2)
            return True
        else:
            raise ValueError(f"Marker with ID {self.stag_id} not found.")

    def homogenize_image_based_on_corners(self):
        if self.corners is None:
            raise ValueError("Corners not detected.")
        x, y, w, h = cv2.boundingRect(self.corners.astype(np.float32))
        destination_corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype='float32')
        transform_matrix = cv2.getPerspectiveTransform(self.corners, destination_corners)
        self.homogenized_image = cv2.warpPerspective(self.image, transform_matrix, (self.image.shape[1], self.image.shape[0]))
        self.save_features_separated(self.homogenized_image, "homogenized")

    def create_mask(self):
        hsv_image = cv2.cvtColor(self.homogenized_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, (0, 0, 120), (180, 255, 255))
        self.save_features_separated(mask, "mask")
        return mask

    def find_and_draw_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.image_with_contours = cv2.drawContours(self.homogenized_image.copy(), contours, -1, (0, 255, 0), 3)
        self.save_features_separated(self.image_with_contours, "contours")
        return contours

    def display_scan_areas_by_markers(self):
        for id, corners in self.scan_areas.items():
            x, y, w, h = corners
            cv2.rectangle(self.homogenized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.save_features_separated(self.homogenized_image, "scan_area")

    def crop_scan_area(self):
        if self.stag_id not in self.scan_areas:
            raise ValueError(f"Scan area for ID {self.stag_id} not defined.")
        x, y, w, h = self.scan_areas[self.stag_id]
        cropped_image = self.homogenized_image[y:y+h, x:x+w]
        self.save_features_separated(cropped_image, "cropped")
        return cropped_image

    def remove_background(self, cropped_image):
        buffer = cv2.imencode('.png', cropped_image)[1].tobytes()
        removed_bg_image = remove(buffer)
        final_image = cv2.imdecode(np.frombuffer(removed_bg_image, np.uint8), cv2.IMREAD_UNCHANGED)
        self.save_features_separated(final_image, "template")

    def save_features_separated(self, img, feature_name):
        directory = os.path.join(self.output_directory, feature_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = f"{feature_name}_{self.file_counter}.png"
        self.file_counter += 1
        cv2.imwrite(os.path.join(directory, filename), img)

if __name__ == "__main__":
    image_path = "./frames/img_0_010.jpg"
    stag_id = 0
    processor = ExtractFeatures(image_path, stag_id)
    if processor.detect_stag():
        processor.homogenize_image_based_on_corners()
        mask = processor.create_mask()
        contours = processor.find_and_draw_contours(mask)
        processor.display_scan_areas_by_markers()
        cropped_image = processor.crop_scan_area()
        processor.remove_background(cropped_image)
        # Continuar com as outras operações conforme necessário
