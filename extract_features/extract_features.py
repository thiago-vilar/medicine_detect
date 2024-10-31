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

        ref_corners = self.corners[0].reshape(-1, 2)
        marker_size_in_mm = 20
        scale_factor = 1
        width, height = marker_size_in_mm * scale_factor, marker_size_in_mm * scale_factor
        dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')
        M = cv2.getPerspectiveTransform(ref_corners.astype('float32'), dst_points)
        self.homogenized_image = cv2.warpPerspective(self.image, M, (width, height))

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

    def crop_scan_area(self):
        if self.selected_id not in self.scan_areas:
            print(f'ID {self.selected_id} não encontrado.')
            return None
        x_min, x_max, y_min, y_max = self.scan_areas[self.selected_id]
        return self.image[y_min:y_max, x_min:x_max]
    
    # def normalized_cropped(self, image):


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
        upper_bound = np.array([250, 250, 250])
        return cv2.inRange(img, lower_bound, upper_bound)
    
 

    def extract_and_draw_contours(self, img, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img_with_contours = img.copy()
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 1)
        plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
        plt.title('Imagem com Contorno Externo (Largest)')
        plt.axis('off')
        plt.show()
        return contours

    def calculate_vectors_and_return_chain_code(self, contour):
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

    def homogenize_image_based_on_stag_orientation(self, image):

        pass

    # def process(self):
    #     if not self.detect_and_label_stags():
    #         return
    #     if not self.display_markers():
    #         return

    #     cropped_image = self.crop_scan_area()
    #     if cropped_image is None:
    #         return

    #     cropped_image_with_no_bg = self.remove_background(cropped_image)
    #     if cropped_image_with_no_bg is None:
    #         return

    #     mask = self.create_mask(cropped_image_with_no_bg)
    #     contours = self.extract_and_draw_contours(cropped_image_with_no_bg, mask)
    #     if not contours:
    #         return

    #     contours_chain_codes = [self.calculate_vectors_and_return_chain_code(contour) for contour in contours]
    #     self.save_features(cropped_image, cropped_image_with_no_bg, mask, contours, contours_chain_codes)

    #     print("Processamento concluído com sucesso.")

if __name__ == "__main__":
    
    image_path = ".\\frames\\img_0_010.jpg"
    selected_id = 0
    processor_image = ExtractFeatures(image_path, selected_id)

    if processor_image.detect_and_label_stags():
        processed_image, ids, corners, scan_area = processor_image.display_markers()
        if processed_image is not None:
            plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))  
            plt.show()
        else:
            print("Não foi possível processar e exibir os marcadores.")
    else:
        print("Falha ao detectar stags na imagem.")
