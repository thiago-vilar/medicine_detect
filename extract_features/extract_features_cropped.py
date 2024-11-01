import cv2
import numpy as np
import matplotlib.pyplot as plt
import stag
from rembg import remove

class ExtractFeatures:
    def __init__(self, image_path, stag_id):
        self.image_path = image_path
        self.stag_id = stag_id
        self.image = None
        self.corners = None
        self.ids = None
        self.homogenized_image = None
        self.scan_areas = {}

    def detect_stag(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            print("Error loading image.")
            return False
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
            print("Corners not detected.")
            return False
        x, y, w, h = cv2.boundingRect(self.corners.astype(np.float32))
        aligned_corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype='float32')

        transform_matrix = cv2.getPerspectiveTransform(self.corners, aligned_corners)
        max_width = self.image.shape[1]
        max_height = self.image.shape[0]
        self.homogenized_image = cv2.warpPerspective(self.image, transform_matrix, (max_width, max_height))
        return True

    def display_markers(self):
        if self.homogenized_image is None:
            print("Homogenized image is not available.")
            return False
        
        if self.corners is None:
            print("No corners available for processing.")
            return False
        
        corner = self.corners.reshape(-1, 2).astype(int)
        centroid_x = int(np.mean(corner[:, 0]))
        centroid_y = int(np.mean(corner[:, 1]))

        cv2.putText(self.homogenized_image, f'ID:{self.stag_id}', (centroid_x + 45, centroid_y -15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

        width = np.max(corner[:, 0]) - np.min(corner[:, 0])
        pixel_size_mm = width / 20
        crop_width = int(25 * pixel_size_mm)
        crop_height = int(75 * pixel_size_mm)
        crop_y_adjustment = int(10 * pixel_size_mm)

        x_min = max(centroid_x - crop_width, 0)
        x_max = min(centroid_x + crop_width, self.homogenized_image.shape[1])
        y_min = max(centroid_y - crop_height - crop_y_adjustment, 0)
        y_max = max(centroid_y - crop_y_adjustment, 0)

        cv2.rectangle(self.homogenized_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv2.putText(self.homogenized_image, 'Scan Area', (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        self.scan_areas[self.stag_id] = (x_min, x_max, y_min, y_max)

        return True
    
    def crop_scan_area(self):
        if self.stag_id not in self.scan_areas:
                print(f'ID {self.stag_id} não encontrado.')
                return None
        x_min, x_max, y_min, y_max = self.scan_areas[self.stag_id]
        return self.homogenized_image[y_min:y_max, x_min:x_max]
    
    def remove_background(self, image_np_array):
        is_success, buffer = cv2.imencode(".jpg", image_np_array)
        if not is_success:
            raise ValueError("Falha ao codificar a imagem para remoção de fundo.")
        output_image = remove(buffer.tobytes())
        img = cv2.imdecode(np.frombuffer(output_image, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Falha ao decodificar a imagem processada.")
        return img

    def create_mask(self, img):
        if img.shape[2] == 4:
            img = img[:, :, :3]
        lower_bound = np.array([30, 30, 30])
        upper_bound = np.array([255, 255, 255])
        return cv2.inRange(img, lower_bound, upper_bound)

if __name__ == "__main__":
    image_path = "./frames/img_0_010.jpg"
    stag_id = 0
    processor_image = ExtractFeatures(image_path, stag_id)
    if processor_image.detect_stag() and processor_image.homogenize_image_based_on_corners():
        if processor_image.display_markers():
            cropped_img =  processor_image.crop_scan_area()
            obj_rm_bg = processor_image.remove_background(cropped_img)
            mask = processor_image.create_mask(obj_rm_bg)

            plt.figure()
            plt.imshow(cv2.cvtColor(processor_image.homogenized_image, cv2.COLOR_BGR2RGB))
            plt.figure()
            plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            plt.show()
            plt.figure()
            plt.imshow(cv2.cvtColor(obj_rm_bg, cv2.COLOR_BGR2RGB))
            plt.show()
            plt.figure()
            plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
            plt.show()
        else:
            print("Failed to display markers.")
    else:
        print("Failed to process image.")
