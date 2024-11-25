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

    def calculate_histograms(self, img_med):
        "Calculates and returns RGB color histograms of an image, excluding any transparent pixels defined by the alpha channel."
        histograms = {}
        if img_med.shape[2] == 4: 
            alpha_mask = img_med[:, :, 3] > 10  
            img_bgr = img_med[alpha_mask, :3]  
            img_rgb = cv2.cvtColor(img_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB)

            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
                histograms[color] = hist.flatten()  
            # # Save
            # directory = 'features/histogram'
            # if not os.path.exists(directory):
            #     os.makedirs(directory)
            # file_number = 0
            # file_path = os.path.join(directory, f'histogram_{file_number}.pkl')
            # while os.path.exists(file_path):
            #     file_number += 1
            #     file_path = os.path.join(directory, f'histogram_{file_number}.pkl')
            # with open(file_path, 'wb') as file:
            #     pickle.dump(histograms, file)
            # print(f"Histograms saved to {file_path}")
        return histograms

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

def load_mask(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def calculate_iou(mask1, mask2):
    mask2_resized = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]))
    intersection = np.logical_and(mask1, mask2_resized)
    union = np.logical_or(mask1, mask2_resized)
    return np.sum(intersection) / np.sum(union)

def display_image(image, title='Image', cmap=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(image if cmap is None else image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    image_path = ".\\frames\\thiago_fotos_10_recognition_afternoon\\img_0_010.jpg"
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

                        mask1 = processor.create_mask(background_removed)
                        plt.imshow(mask1, cmap='gray')
                        plt.title('Mask 1')
                        plt.show()

        mask_file_path = '.\\features\\mask\\mask_0.pkl'
        mask2 = load_mask(mask_file_path)
        display_image(mask2, 'Mask 2 from .pkl', 'gray')
        print(mask1.shape) 
        print(mask2.shape)

        iou = calculate_iou(mask1, mask2)
        print(f'IoU Score: {iou:.2f}')
        if iou == 1.0 or iou >= 0.7:
            print('The masks are identical.')
        elif iou < 0.7 and iou >=0.4:
            print('The mask has low similarity')
        else:
            print('The masks are not similar.')
