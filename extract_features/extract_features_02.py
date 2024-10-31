import cv2
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
import os
import stag
import pickle

class ExtractFeatures:
    def __init__(self, image_path, selected_id):
        """Initialize with image path and ID of the STAG marker to focus on."""
        self.image_path = image_path
        self.selected_id = selected_id
        self.base_path = "features"
        self.image = None
        self.corners = None
        self.ids = None
        self.scan_areas = {}

    def detect_and_label_stags(self):
        """Load the image and detect STAG markers, storing detected corners and IDs."""
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            print("Erro ao carregar a imagem.")
            return False
        
        config = {'libraryHD': 17, 'errorCorrection': None}
        self.corners, self.ids, _ = stag.detectMarkers(self.image, **config)
        if self.ids is None:
            print("Nenhum marcador foi encontrado.")
            return False
        return True

    def display_markers(self):
        """Add visual markers for detected STAGs and define scanning areas based on their locations."""
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

            x_min = max(centroid_x - crop_width // 2, 0)
            x_max = min(centroid_x + crop_height // 2, self.image.shape[1])
            y_min = max(centroid_y - crop_width - crop_y_adjustment, 0)
            y_max = max(centroid_y - crop_y_adjustment, 0)

            cv2.rectangle(self.image, (x_min, y_min), (x_max, y_max), (255, 0, 255), 1)
            cv2.putText(self.image, 'ScanArea', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)

            self.scan_areas[id_] = (x_min, x_max, y_min, y_max)

        return True

    def crop_scan_area(self):
        """Crop the image based on the scanning area defined by the selected ID."""
        if self.selected_id not in self.scan_areas:
            print(f'ID {self.selected_id} não encontrado.')
            return None
        x_min, x_max, y_min, y_max = self.scan_areas[self.selected_id]
        return self.image[y_min:y_max, x_min:x_max]

    def remove_background(self, image_np_array):
        """Remove the background from the cropped image, retaining only the object of interest."""
        is_success, buffer = cv2.imencode(".jpg", image_np_array)
        if not is_success:
            raise ValueError("Falha ao codificar a imagem para remoção de fundo.")
        output_image = remove(buffer.tobytes())
        img = cv2.imdecode(np.frombuffer(output_image, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Falha ao decodificar a imagem processada.")
        return img

    def create_mask(self, img):
        """Create a binary mask for the image based on a specified color range."""
        if img.shape[2] == 4:
            img = img[:, :, :3]
        lower_bound = np.array([30, 30, 30])
        upper_bound = np.array([250, 250, 250])
        return cv2.inRange(img, lower_bound, upper_bound)

    def extract_and_draw_contours(self, img, mask):
        """Extract and draw contours on the image based on the provided mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img_with_contours = img.copy()
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 1)
        plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
        plt.title('Imagem com Contorno Externo (Largest)')
        plt.axis('off')
        plt.show()
        return contours

    def calculate_vectors_and_return_chain_code(self, contour):
        """Calculate and return the chain code of a contour, representing the sequence of directions in the contour."""
        directions = np.array([(1, 0), (1, -1), (0, -1), (-1, -1), 
                               (-1, 0), (-1, 1), (0, 1), (1, 1)])
        chain_code = []
        for i in range(1, len(contour)):
            diff = np.array([contour[i][0][0] - contour[i-1][0][0], contour[i][0][1] - contour[i-1][0][1]])
            if np.linalg.norm(diff) == 0:
                continue
            norm_diff = diff / np.linalg.norm(diff)
            distances = np.linalg.norm(directions - norm_diff, axis=1)
            closest_direction_idx = np.argmin(distances)
            chain_code.append(closest_direction_idx)
        return chain_code

    def save_features(self, image_cropped, obj_png, mask, contours_matrix, contours_chain_code):
        """Save the extracted features in specific folders within a base folder."""
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

        feature_types = {
            'image_cropped': image_cropped,
            'obj_png': obj_png,
            'mask': mask,
            'contours_matrix': contours_matrix,
            'contours_chain_code': contours_chain_code
        }

        for feature_name, feature_data in feature_types.items():
            feature_folder = os.path.join(self.base_path, feature_name)
            if not os.path.exists(feature_folder):
                os.makedirs(feature_folder)

            existing_files = len([name for name in os.listdir(feature_folder) if os.path.isfile(os.path.join(feature_folder, name))])
            file_name = f"{feature_name}_{str(existing_files + 1).zfill(3)}.pkl"
            file_path = os.path.join(feature_folder, file_name)

            with open(file_path, 'wb') as file:
                pickle.dump(feature_data, file)
            print(f"Feature '{feature_name}' saved as {file_path}")

    def homogenize_image_based_on_stag_orientation(self, image, corners):
        """Adjust the image orientation based on the orientation of the detected STAG marker."""
        if corners is None:
            return image

        # Calculate rotation angle of the marker
        vector = corners[0][1] - corners[0][0]  # Assumes that points 0 and 1 of the marker define the horizontal orientation
        angle = np.arctan2(vector[1], vector[0])
        degrees = np.degrees(angle)

        # Calculate the center of rotation
        center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, -degrees, 1.0)

        # Rotate the image to normalize the orientation of the marker
        rotated_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        return rotated_image

    def process(self):
        """Execute the complete image processing from detection to homogenization and feature saving."""
        if not self.detect_and_label_stags():
            return
        if not self.display_markers():
            return

        cropped_image = self.crop_scan_area()
        if cropped_image is None:
            return

        # Homogenize image orientation based on the STAG marker orientation
        if self.corners is not None and self.selected_id in self.scan_areas:
            selected_corners = self.corners[np.where(self.ids.flatten() == self.selected_id)[0][0]]
            cropped_image = self.homogenize_image_based_on_stag_orientation(cropped_image, selected_corners)

        cropped_image_with_no_bg = self.remove_background(cropped_image)
        if cropped_image_with_no_bg is None:
            return

        mask = self.create_mask(cropped_image_with_no_bg)
        contours = self.extract_and_draw_contours(cropped_image_with_no_bg, mask)
        if not contours:
            return

        contours_chain_codes = [self.calculate_vectors_and_return_chain_code(contour) for contour in contours]
        self.save_features(cropped_image, cropped_image_with_no_bg, mask, contours, contours_chain_codes)

        print("Processamento concluído com sucesso.")

if __name__ == "__main__":
    image_path = "frames\img_0_010.jpg"
    selected_id = 0
    processor = ExtractFeatures(image_path, selected_id)
    processor.process()
