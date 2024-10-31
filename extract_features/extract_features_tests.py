import cv2
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
import os
import stag
import pickle

class ExtractFeatures:
    def __init__(self, image_path, selected_id):
        self.image_path = image_path
        self.selected_id = selected_id
        self.base_path = "features"
        self.image = None
        self.corners = None
        self.ids = None
        self.scan_areas = {}
        self.homogenized_image = None

    def detect_and_label_stags(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            print("Erro ao carregar a imagem.")
            return False
        config = {'libraryHD': 17, 'errorCorrection': -1}
        self.corners, self.ids, _ = stag.detectMarkers(self.image, **config)
        if self.ids is None:
            print("Nenhum marcador foi encontrado.")
            return False
        return True

    def homogenize_image_based_on_stag_orientation(self):
        if self.corners is None or self.ids is None or len(self.corners) < 1:
            print("Não há marcadores suficientes para homogeneização.")
            return False

        object_points = np.array([
            [-10, 10, 0],  # marker 20mm
            [10, 10, 0],
            [10, -10, 0],
            [-10, -10, 0]
        ], dtype=np.float32)

        transformed_image = np.copy(self.image)
        
        for corners in self.corners:
            image_points = corners[0].astype(np.float32)

            camera_matrix = np.array([[self.image.shape[1]/2, 0, self.image.shape[1]/2],
                                    [0, self.image.shape[0]/2, self.image.shape[0]/2],
                                    [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.zeros(4)

            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
            if not success:
                print("Não foi possível resolver a pose do marcador.")
                continue

            rot_mat, _ = cv2.Rodrigues(rvec)
            proj_matrix = np.hstack((rot_mat, tvec))
            homography_matrix = np.dot(camera_matrix, proj_matrix)
            homography_matrix = homography_matrix[:, :3]

            transformed_image = cv2.warpPerspective(transformed_image, homography_matrix, (self.image.shape[1], self.image.shape[0]))

        self.homogenized_image = transformed_image
        return True

    def display_markers(self):
        if self.corners is None or self.ids is None:
            return False

        for corners, id_ in zip(self.corners, self.ids.flatten()):
            corner = corners.reshape(-1, 2).astype(int)
            centroid_x = int(np.mean(corner[:, 0]))
            centroid_y = int(np.mean(corner[:, 1]))
            cv2.polylines(self.image, [corner], True, (255, 0, 255), 1)
            cv2.putText(self.image, f'ID: {id_}', (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 1)

            width = np.max(corner[:, 0]) - np.min(corner[:, 0])
            pixel_size_mm = width / 20  
            crop_width = int(75 * pixel_size_mm)
            crop_height = int(50 * pixel_size_mm)  
            crop_y_adjustment = int(15 * pixel_size_mm)

            x_min = max(centroid_x - crop_height // 2, 0)
            x_max = min(centroid_x + crop_height // 2, self.image.shape[1])
            y_min = max(centroid_y - crop_width - crop_y_adjustment, 0)
            y_max = max(centroid_y - crop_y_adjustment, 0)

            cv2.rectangle(self.image, (x_min, y_min), (x_max, y_max), (255, 0, 255), 1)
            cv2.putText(self.image, 'Scan Area', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)

            self.scan_areas[id_] = (x_min, x_max, y_min, y_max)

        return self.image, self.ids, self.corners, self.scan_areas

if __name__ == "__main__":
    image_path = ".\\frames\\img_0_010.jpg"
    selected_id = 0
    processor_image = ExtractFeatures(image_path, selected_id)
    if processor_image.detect_and_label_stags():
        if processor_image.homogenize_image_based_on_stag_orientation():
            image, ids, corners, scan_area = processor_image.display_markers()
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
