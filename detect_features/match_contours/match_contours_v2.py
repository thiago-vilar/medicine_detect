import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import stag
from rembg import remove
import pickle

class ExtractFeatures:
    ''' Initializes with the path to an image and a specific marker (stag) ID. '''
    def __init__(self, image_path, stag_id):
        self.image_path = image_path
        self.stag_id = stag_id
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError("Image could not be loaded.")
        self.corners = None
        self.ids = None
        self.homogenized_image = None
        self.scan_areas = {}
        self.pixel_size_mm = None 

    ''' Detects a predefined stag marker in the image using the stag library. '''
    def detect_stag(self):
        config = {'libraryHD': 17, 'errorCorrection': -1}
        self.corners, self.ids, _ = stag.detectMarkers(self.image, **config)
        if self.ids is not None and self.stag_id in self.ids:
            index = np.where(self.ids == self.stag_id)[0][0]
            self.corners = self.corners[index].reshape(-1, 2)
            self.calculate_pixel_size_mm()
            return True
        print("Marker with ID", self.stag_id, "not found.")
        return False

    ''' Calculates the pixel size in millimeters based on the detected stag marker. '''
    def calculate_pixel_size_mm(self):
        if self.corners is not None:
            width_px = np.max(self.corners[:, 0]) - np.min(self.corners[:, 0])
            self.pixel_size_mm = 20.0 / width_px 

    ''' Normalizes the image perspective based on detected stag corners. '''
    def homogenize_image_based_on_corners(self):
        if self.corners is None:
            print("Corners not detected.")
            return None
        x, y, w, h = cv2.boundingRect(self.corners.astype(np.float32))
        aligned_corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype='float32')
        transform_matrix = cv2.getPerspectiveTransform(self.corners, aligned_corners)
        self.homogenized_image = cv2.warpPerspective(self.image, transform_matrix, (self.image.shape[1], self.image.shape[0]))
        return self.homogenized_image

    ''' Displays the scan area on the homogenized image based on the stag location. '''
    def display_scan_area_by_markers(self):
        if self.homogenized_image is None:
            print("Homogenized image is not available.")
            return None
        
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
        self.scan_areas[self.stag_id] = (x_min, x_max, y_min, y_max)
        return self.homogenized_image

    def crop_scan_area(self):
        """Crops the defined scan area from the homogenized image and saves it locally."""
        if self.stag_id not in self.scan_areas:
            print(f'ID {self.stag_id} not found.')
            return None
        x_min, x_max, y_min, y_max = self.scan_areas[self.stag_id]
        cropped_image = self.homogenized_image[y_min:y_max, x_min:x_max]
        # # Save
        # directory_path = 'features/cropped_imgs'
        # if not os.path.exists(directory_path):
        #     os.makedirs(directory_path)    
        # file_number = 0
        # while os.path.exists(f'{directory_path}/img_cropped_{file_number}.png'):
        #     file_number += 1
        # file_path = f'{directory_path}/img_cropped_{file_number}.png'
        # cv2.imwrite(file_path, cropped_image)
        # print(f'Image saved as {file_path}')
        return cropped_image


    def remove_background(self, image_np_array):
        """Removes the background from the cropped scan area and saves the image with alpha channel."""
        is_success, buffer = cv2.imencode(".jpg", image_np_array)
        if not is_success:
            raise ValueError("Failed to encode image for background removal.")
        output_image = remove(buffer.tobytes())
        img_med = cv2.imdecode(np.frombuffer(output_image, np.uint8), cv2.IMREAD_UNCHANGED)
        if img_med is None:
            raise ValueError("Failed to decode processed image.")
        # # Save
        # if not os.path.exists('features/medicine_png'):
        #     os.makedirs('features/medicine_png')
        # file_number = 0
        # while os.path.exists(f'features/medicine_png/medicine_{file_number}.png'):
        #     file_number += 1
        # cv2.imwrite(f'features/medicine_png/medicine_{file_number}.png', img_med)
        # print(f'Image saved as medicine_{file_number}.png with transparent background')
        return img_med

    def create_mask(self, img):
        """Creates a binary mask for the foreground object in the image and saves it with transparency."""
        if img.shape[2] == 4:
            img = img[:, :, :3]  # Remove alpha channel
        lower_bound = np.array([30, 30, 30])
        upper_bound = np.array([255, 255, 255])
        mask = cv2.inRange(img, lower_bound, upper_bound)
        # Convert binary mask to 4-channel 
        mask_rgba = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGBA)
        mask_rgba[:, :, 3] = mask 

        # # Save the mask as a .pkl file
        # directory = 'features/mask'
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # file_number = 0
        # while os.path.exists(f'{directory}/mask_{file_number}.pkl'):
        #     file_number += 1
        # file_path = f'{directory}/mask_{file_number}.pkl'
        # with open(file_path, 'wb') as file:
        #     pickle.dump(mask, file)
        # print(f'Mask saved as {file_path} with transparency in {directory}')

        return mask

    def find_and_draw_contours(self, mask):
        """Finds and draws only the largest contour around the foreground object based on the mask and saves the image with alpha transparency and the contour data in a .pkl file."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if largest_contour.size > 0:
                mask_with_contours = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
                mask_with_contours[:, :, 3] = mask
                cv2.drawContours(mask_with_contours, [largest_contour], -1, (0, 0, 255, 255), 2)

                # # Save images and contours
                # directory = 'features/contours'
                # if not os.path.exists(directory):
                #     os.makedirs(directory)
                # file_number = 0
                # while os.path.exists(f'{directory}/contour_{file_number}.png'):
                #     file_number += 1

                # # Saving the image
                # image_path = f'{directory}/contour_{file_number}.png'
                # cv2.imwrite(image_path, mask_with_contours)
                # print(f'Contour image saved as {image_path}')

                # # Saving the contour as a pickle file
                # pkl_path = f'{directory}/contour_{file_number}.pkl'
                # with open(pkl_path, 'wb') as file:
                #     pickle.dump(largest_contour, file)
                # print(f'Contour data saved as {pkl_path}')

                return mask_with_contours, largest_contour
        else:
            return None

    def compute_chain_code(self, contour):
        ''' Calculates chain code for object contours for shape analysis. '''
        start_point = contour[0][0]
        current_point = start_point
        chain_code = []
        moves = {
            (-1, 0) : 3,
            (-1, 1) : 2,
            (0, 1)  : 1,
            (1, 1)  : 0,
            (1, 0)  : 7,
            (1, -1) : 6,
            (0, -1) : 5,
            (-1, -1): 4
        }
        for i in range(1, len(contour)):
            next_point = contour[i][0]
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            if dx != 0:
                dx = dx // abs(dx)
            if dy != 0:
                dy = dy // abs(dy)
            move = (dx, dy)
            if move in moves:
                chain_code.append(moves[move])
            current_point = next_point
        # Close the loop
        dx = start_point[0] - current_point[0]
        dy = start_point[1] - current_point[1]
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)
        move = (dx, dy)
        if move in moves:
            chain_code.append(moves[move])
        # # Save
        # directory = 'features/chain_code'
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # file_number = 0
        # file_path = os.path.join(directory, f'chain_code_{file_number}.pkl')
        # while os.path.exists(file_path):
        #     file_number += 1
        #     file_path = os.path.join(directory, f'chain_code_{file_number}.pkl')
        
        # with open(file_path, 'wb') as file:
        #     pickle.dump(chain_code, file)
        # print(f"Chain code saved to {file_path}")
        # print("Chain code sequence:", chain_code)

        return chain_code, len(chain_code)


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

def calculate_contour_iou(contour1, contour2, shape):
    mask1 = np.zeros(shape, dtype=np.uint8)
    mask2 = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask1, [contour1], -1, 255, -1)
    cv2.drawContours(mask2, [contour2], -1, 255, -1)
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)


def calculate_chain_code_iou(chain_code1, chain_code2):
    # Verifique se os tamanhos são iguais
    min_length = min(len(chain_code1), len(chain_code2))
    chain_code1 = chain_code1[:min_length]
    chain_code2 = chain_code2[:min_length]
    intersection = np.sum(np.array(chain_code1) == np.array(chain_code2))
    union = len(chain_code1)  # Como ambos os códigos têm o mesmo comprimento
    return intersection / union


if __name__ == "__main__":
    image_path = ".\\frames\\thiago_fotos_10_feature_afternoon\\img_0_009.jpg"
    stag_id = 0
    processor = ExtractFeatures(image_path, stag_id)

    if processor.detect_stag():
        homogenized = processor.homogenize_image_based_on_corners()
        if homogenized is not None:
            plt.imshow(cv2.cvtColor(homogenized, cv2.COLOR_BGR2RGB))
            plt.title('Homogenized Image')
            plt.show()

            marked_image = processor.display_scan_area_by_markers()
            if marked_image is not None:
                plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
                plt.title('Marked Scan Area')
                plt.show()

                cropped = processor.crop_scan_area()
                if cropped is not None:
                    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    plt.title('Cropped Scan Area')
                    plt.show()

                    background_removed = processor.remove_background(cropped)
                    if background_removed is not None:
                        img_med = background_removed.copy()
                        plt.imshow(cv2.cvtColor(img_med, cv2.COLOR_BGR2RGB))
                        plt.title('Arq.png - Background Removed')
                        plt.show()

                        mask = processor.create_mask(background_removed)
                        plt.imshow(mask, cmap='gray')
                        plt.title('Mask')
                        plt.show()

                        contoured_image, largest_contour = processor.find_and_draw_contours(mask)
                        if largest_contour is not None:
                            chain_code, _ = processor.compute_chain_code(largest_contour)

                            # Carregar contornos e chain codes salvos
                            loader = LoadContours('features/contours/contour_0.pkl', 'features/chain_code/chain_code_1.pkl')
                            loaded_contour, loaded_chain_code = loader.load_data()

                            # Comparar Contornos
                            shape = mask.shape
                            iou_contour = calculate_contour_iou(largest_contour, loaded_contour, shape)
                            print(f"IoU Score for Contour: {iou_contour:.2f}")

                            # Exibir os contornos extraído e carregado
                            blank_image = np.zeros(mask.shape, dtype=np.uint8)
                            cv2.drawContours(blank_image, [largest_contour], -1, 255, 2)
                            plt.imshow(blank_image, cmap='gray')
                            plt.title('Extracted Contour')
                            plt.show()

                            blank_image_loaded = np.zeros(mask.shape, dtype=np.uint8)
                            cv2.drawContours(blank_image_loaded, [loaded_contour], -1, 255, 2)
                            plt.imshow(blank_image_loaded, cmap='gray')
                            plt.title('Loaded Contour')
                            plt.show()

                            # Comparar Chain Codes
                            iou_chain_code = calculate_chain_code_iou(chain_code, loaded_chain_code)
                            print(f"IoU Score for Chain Code: {iou_chain_code:.2f}")

                            # Exibir os chain codes extraído e carregado
                            print("Extracted Chain Code:", chain_code)
                            print("Loaded Chain Code:", loaded_chain_code)

                            # Avaliação do IoU para contornos
                            if iou_contour == 1.0:
                                print("The contours are identical.")
                            elif iou_contour >= 0.7:
                                print("The contours are highly similar.")
                            elif iou_contour >= 0.4:
                                print("The contours have low similarity.")
                            else:
                                print("The contours are not similar.")
