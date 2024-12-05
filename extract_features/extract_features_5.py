import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import stag
from rembg import remove
import pickle
from skimage import io, img_as_float
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour

class ExtractFeatures:
    def __init__(self, image_path, stag_id, med_type):
        """Initializes the class with path to the image, an identifier for the stag marker, and the type of medicine."""
        self.image_path = image_path
        self.stag_id = stag_id
        self.med_type = med_type
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError("Image could not be loaded.")
        self.corners = None
        self.ids = None
        self.homogenized_image = None
        self.scan_areas = {}
        self.pixel_size_mm = None
        self.data = []

    def detect_stag(self):
        """Detects a specific stag marker in the image and initializes measurements for pixel size."""
        config = {'libraryHD': 17, 'errorCorrection': -1}
        self.corners, self.ids, _ = stag.detectMarkers(self.image, **config)
        if self.ids is not None and self.stag_id in self.ids:
            index = np.where(self.ids == self.stag_id)[0][0]
            self.corners = self.corners[index].reshape(-1, 2)
            self.calculate_pixel_size_mm()
            return True
        print("Marker with ID", self.stag_id, "not found.")
        return False

    def calculate_pixel_size_mm(self):
        """Calculates the pixel size in millimeters using the stag marker as reference."""
        if self.corners is not None:
            width_px = np.max(self.corners[:, 0]) - np.min(self.corners[:, 0])
            self.pixel_size_mm = 20.0 / width_px 

    def homogenize_image_based_on_corners(self):
        """Normalizes the image perspective based on the detected stag corners."""
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

    def display_scan_area_by_markers(self, med_type):
        """Displays the scan area on the homogenized image based on the stag location."""
        if self.homogenized_image is None:
            print("Homogenized image is not available.")
            return None
        corner = self.corners.reshape(-1, 2).astype(int)
        centroid_x = int(np.mean(corner[:, 0]))
        centroid_y = int(np.mean(corner[:, 1]))
        cv2.putText(self.homogenized_image, f'ID:{self.stag_id}', (centroid_x + 45, centroid_y -15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        if med_type == "Pill":  
            width = np.max(corner[:, 0]) - np.min(corner[:, 0])
            pixel_size_mm = width / 20
            crop_width = int(30 * pixel_size_mm)
            crop_height = int(40 * pixel_size_mm)
            crop_y_adjustment = int(10 * pixel_size_mm)
            x_min = max(centroid_x - crop_width, 0)
            x_max = min(centroid_x + crop_width, self.homogenized_image.shape[1])
            y_min = max(centroid_y - crop_height - crop_y_adjustment, 0)
            y_max = max(centroid_y - crop_y_adjustment, 0)
            cv2.rectangle(self.homogenized_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            self.scan_areas[self.stag_id] = (x_min, x_max, y_min, y_max)
            return self.homogenized_image
        elif med_type == "Ampoule":  
            width = np.max(corner[:, 0]) - np.min(corner[:, 0])
            pixel_size_mm = width / 20
            crop_width = int(8 * pixel_size_mm)
            crop_height = int(50 * pixel_size_mm)
            crop_y_adjustment = int(10 * pixel_size_mm)
            x_min = max(centroid_x - crop_width, 0)
            x_max = min(centroid_x + crop_width, self.homogenized_image.shape[1])
            y_min = max(centroid_y - crop_height - crop_y_adjustment, 0)
            y_max = max(centroid_y - crop_y_adjustment, 0)
            cv2.rectangle(self.homogenized_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            self.scan_areas[self.stag_id] = (x_min, x_max, y_min, y_max)
            return self.homogenized_image
        elif med_type == "Flask":
            width = np.max(corner[:, 0]) - np.min(corner[:, 0])
            pixel_size_mm = width / 20
            crop_width = int(20* pixel_size_mm)
            crop_height = int(40 * pixel_size_mm)
            crop_y_adjustment = int(10 * pixel_size_mm)
            x_min = max(centroid_x - crop_width, 0)
            x_max = min(centroid_x + crop_width, self.homogenized_image.shape[1])
            y_min = max(centroid_y - crop_height - crop_y_adjustment, 0)
            y_max = max(centroid_y - crop_y_adjustment, 0)
            cv2.rectangle(self.homogenized_image, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
            self.scan_areas[self.stag_id] = (x_min, x_max, y_min, y_max)
            return self.homogenized_image
        else:
            width = np.max(corner[:, 0]) - np.min(corner[:, 0])
            pixel_size_mm = width /20
            crop_width = int(30*pixel_size_mm)
            crop_height = int(75*pixel_size_mm)
            crop_y_adjustment = int(10 * pixel_size_mm)
            x_min = max(centroid_x - crop_width, 0)
            x_max = min(centroid_x + crop_width, self.homogenized_image.shape[1])
            y_min = max(centroid_y - crop_height - crop_y_adjustment, 0)
            y_max = max(centroid_y - crop_y_adjustment, 0)
            cv2.rectangle(self.homogenized_image, (x_min, y_min), (x_max, y_max), (0,255,0),1)
            self.scan_areas[self.stag_id] = (x_min, x_max, y_min, y_max)
            return self.homogenized_image

    def crop_scan_area(self):
        """Crops the defined scan area from the homogenized image for further processing."""
        if self.stag_id not in self.scan_areas:
            print(f'ID {self.stag_id} not found.')
            return None
        x_min, x_max, y_min, y_max = self.scan_areas[self.stag_id]
        cropped_image = self.homogenized_image[y_min:y_max, x_min:x_max]
        # Save
        directory_path = 'features/cropped_imgs'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)    
        file_number = 0
        while os.path.exists(f'{directory_path}/img_cropped_{file_number}.png'):
            file_number += 1
        file_path = f'{directory_path}/img_cropped_{file_number}.png'
        cv2.imwrite(file_path, cropped_image)
        print(f'Image saved as {file_path}')
        return cropped_image

    def filter_pdi(self, image):
        """Applies Gaussian blur followed by a Sobel filter to enhance horizontal edges of a reflective object."""
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # img_blur = cv2.GaussianBlur(img_gray, (1, 1), 0)
        filter = np.array([
            [ 0, -1, 0],
            [-1, 4, -1],
            [ 0, -1, 0]
        ])
        ddepth = cv2.CV_16S
        img_filtered = cv2.filter2D(img_gray, ddepth, filter)
        abs_img_filtered = cv2.convertScaleAbs(img_filtered)
        return abs_img_filtered

    def apply_blend_filter(self, image):
        """Transforms image to grayscale, applies Sobel filter, and overlays it on the original image."""
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # img_blur = cv2.GaussianBlur(img_gray, (1, 1), 0)
        filter = np.array([
            [ 0, -1, 0],
            [-1, 4, -1],
            [ 0, -1, 0]
        ])
        ddepth = cv2.CV_16S
        img_filtered = cv2.filter2D(img_gray, ddepth, filter)
        abs_img_filtered = cv2.convertScaleAbs(img_filtered)
        alpha = 0.8 
        blended = cv2.addWeighted(image, 1, cv2.cvtColor(abs_img_filtered, cv2.COLOR_GRAY2BGR), alpha, 0)
        return blended

    def remove_background(self, image_np_array):
        """Removes the background from the image using the rembg library."""
        is_success, buffer = cv2.imencode(".png", image_np_array)
        if not is_success:
            raise ValueError("Failed to encode image for background removal.")
        output_image = remove(buffer.tobytes())
        img_med = cv2.imdecode(np.frombuffer(output_image, np.uint8), cv2.IMREAD_UNCHANGED)
        if img_med is None:
            raise ValueError("Failed to decode processed image.")
        return img_med

    def calculate_histograms(self, img_med):
        """Calculates and returns RGB histograms for the processed image."""
        histograms = {}
        if img_med.shape[2] == 4:  
            alpha_mask = img_med[:, :, 3] > 10
            img_bgr = img_med[alpha_mask, :3]
        else:
            img_bgr = img_med
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
            histograms[color] = hist.flatten()
        return histograms


    def create_mask(self, img):
        """Creates a binary mask for the foreground object in the image for segmentation,
        including certain RGB ranges and excluding certain HSV ranges."""
        
        # Se a imagem tiver um canal alfa, remova-o
        if img.shape[2] == 4:
            img = img[:, :, :3]
        # Definir os limites para a máscara RGB (incluindo)
        lower_bound_rgb = np.array([30, 30, 30])
        upper_bound_rgb = np.array([255, 255, 255])
        
        # Criar máscara RGB
        mask_rgb = cv2.inRange(img, lower_bound_rgb, upper_bound_rgb)
        
        # Converter a imagem para o espaço de cor HSV para processamento de exclusão
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Definir os limites para a máscara HSV (excluindo)
        lower_bound_hsv = np.array([0, 4, 94])
        upper_bound_hsv = np.array([43, 255, 255])
      
        # Criar máscara HSV
        mask_hsv = cv2.inRange(img_hsv, lower_bound_hsv, upper_bound_hsv)
        
        # Inverter a máscara HSV para obter a área que não está nos thresholds de exclusão
        mask_hsv_inverted = cv2.bitwise_not(mask_hsv)
        
        # Combinar as máscaras para incluir RGB e excluir áreas HSV
        final_mask = cv2.bitwise_and(mask_rgb, mask_hsv_inverted)
        
        return final_mask


    def find_and_draw_contours(self, mask):
        """Finds and draws the largest contour around the foreground object."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if largest_contour.size > 0:
                mask_with_contours = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
                mask_with_contours[:, :, 3] = mask
                cv2.drawContours(mask_with_contours, [largest_contour], -1, (0, 0, 255, 255), 2)
                return mask_with_contours, largest_contour
        else:
            return None

    def compute_chain_code(self, contour):
        """Calculates the chain code for the contour which can be used for shape analysis."""
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
        return chain_code, len(chain_code)

    def draw_chain_code(self, img_med, contour, chain_code):
        """Draws the chain code on the image to visually represent contour direction changes."""
        start_point = tuple(contour[0][0])
        current_point = start_point
        moves = {
            0: (1, 1),    # bottom-right
            1: (0, 1),    # right
            2: (-1, 1),   # top-right
            3: (-1, 0),   # top
            4: (-1, -1),  # top-left
            5: (0, -1),   # left
            6: (1, -1),   # bottom-left
            7: (1, 0)     # bottom
        }
        for code in chain_code:
            dx, dy = moves[code]
            next_point = (current_point[0] + dx, current_point[1] + dy)
            cv2.line(img_med, current_point, next_point, (255, 255, 255), 1)
            current_point = next_point
        return img_med

    def medicine_measures(self, cropped_img, largest_contour):
        """Measures the dimensions of the detected contours and returns a list of measures."""
        if largest_contour is None or len(largest_contour) == 0:
            print("No contours found.")
            return None
        px_to_mm_scale = self.pixel_size_mm
        measures = []
        measured_img = cropped_img.copy()
        for point in largest_contour:
            x, y, w, h = cv2.boundingRect(point)
            width_mm = w * px_to_mm_scale
            height_mm = h * px_to_mm_scale
            measures.append((width_mm, height_mm))
            cv2.rectangle(measured_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(measured_img, f"{width_mm:.1f}mm x {height_mm:.1f}mm", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return measures, measured_img

    def collect_data(self, img_med, mask, largest_contour, chain_code, measures, histograms):
        """Collects all extracted data and prepares it for saving or further analysis."""
        # Ensure the directory for saving images exists
        images_directory = "extracted_features/images"
        os.makedirs(images_directory, exist_ok=True)
        # Save the median image with background removed
        med_img_path = os.path.join(images_directory, f"{self.stag_id}_img_med.png")
        cv2.imwrite(med_img_path, img_med)
        # Save the mask image
        mask_img_path = os.path.join(images_directory, f"{self.stag_id}_mask.png")
        mask_rgba = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGBA)  # Convert mask to RGBA for visibility
        mask_rgba[:, :, 3] = mask  # Apply the mask to the alpha channel
        cv2.imwrite(mask_img_path, mask_rgba)
        # Assuming you want to save contour drawing
        contour_img_path = os.path.join(images_directory, f"{self.stag_id}_contour.png")
        contour_img = img_med.copy()
        cv2.drawContours(contour_img, [largest_contour], -1, (0, 255, 0), 3)
        cv2.imwrite(contour_img_path, contour_img)
        # Prepare data entry
        data_entry = {
            "Medicine Type": self.med_type,
            "Image Path": med_img_path,
            "Mask Path": mask_img_path,
            "Contour Path": contour_img_path,
            "Chain Code Length": len(chain_code),
            "Chain Code": str(chain_code),
            "Width (mm)": measures[0][0] if measures and len(measures) > 0 else None,
            "Height (mm)": measures[0][1] if measures and len(measures) > 0 else None,
            **{f"{color}_histogram": hist.tolist() for color, hist in histograms.items()}
        }
        self.data.append(data_entry)

    def save_data_to_csv(self, filename='medicines_features.csv'):
        """Appends the processed data to a CSV file, ensuring data continuity."""
        if self.data:
            df = pd.read_csv(filename) if os.path.exists(filename) else pd.DataFrame()
            new_df = pd.DataFrame(self.data)
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(filename, index=False)
            print(f"Data appended to {filename}.")


def main():
    """Main function to execute the process and display results of each step."""
    image_path = ".\\frames\\matte black box\\img_0_004.jpg"
    stag_id = 0
    med_type = "Ampoule"
    processor = ExtractFeatures(image_path, stag_id, med_type)
    
    if processor.detect_stag():
        homogenized = processor.homogenize_image_based_on_corners()
        if homogenized is not None:
            plt.imshow(cv2.cvtColor(homogenized, cv2.COLOR_BGR2RGB))
            plt.title('Homogenized Image')
            plt.show()

            marked_image = processor.display_scan_area_by_markers(med_type)
            if marked_image is not None:
                plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
                plt.title('Marked Scan Area')
                plt.show()

                cropped = processor.crop_scan_area()
                if cropped is not None:
                    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    plt.title('Cropped Scan Area')
                    plt.show()

                    filtered_pdi = processor.filter_pdi(cropped)
                    if filtered_pdi is not None:
                        plt.imshow(filtered_pdi, cmap='gray')
                        plt.title('Experimental Filter')
                        plt.show()

                        # detect_curv = processor.detect_curvature(cropped)
                        # plt.imshow(detect_curv)
                        # plt.title('Detect Curve')
                        # plt.show()

                        # snake_cont = processor.snake_contour(cropped)
                        # plt.imshow(snake_cont, cmap='gray')
                        # plt.title('Snake')
                        # plt.show()

                        blend_filt = processor.apply_blend_filter(cropped)
                        if blend_filt is not None:    
                            plt.imshow(blend_filt, cmap='gray')
                            plt.title('Blend')
                            plt.show()

                            background_removed = processor.remove_background(blend_filt)
                            if background_removed is not None:
                                img_med = background_removed.copy()
                                plt.imshow(cv2.cvtColor(img_med, cv2.COLOR_BGR2RGB))
                                plt.title('Background Removed')
                                plt.show()

                                mask = processor.create_mask(background_removed)
                                if mask is not None:
                                    plt.imshow(mask, cmap='gray')
                                    plt.title('Mask')
                                    plt.show()

                                    contoured_image, largest_contour = processor.find_and_draw_contours(mask)
                                    if contoured_image is not None and largest_contour is not None and largest_contour.size > 0:
                                        plt.imshow(cv2.cvtColor(contoured_image, cv2.COLOR_BGR2RGB))
                                        plt.title('Largest Contour by Mask')
                                        plt.show()

                                        chain_code, _ = processor.compute_chain_code(largest_contour)
                                        if chain_code is not None:
                                            chain_drawn_image = processor.draw_chain_code(img_med, largest_contour, chain_code)
                                            plt.imshow(cv2.cvtColor(chain_drawn_image, cv2.COLOR_BGR2RGB))
                                            plt.title('Chain Code Drawn')
                                            plt.show()

                                            measures, measured_medicine = processor.medicine_measures(img_med, [largest_contour])
                                            if measured_medicine is not None:
                                                plt.imshow(cv2.cvtColor(measured_medicine, cv2.COLOR_BGR2RGB))
                                                plt.title('Measured Medicine')
                                                plt.show()

                                                histograms = processor.calculate_histograms(img_med)
                                                if histograms:
                                                    plt.figure(figsize=(10, 5))
                                                    plt.title('RGB Histograms')
                                                    plt.xlabel('Pixel Intensity')
                                                    plt.ylabel('Pixel Count')
                                                    for color in histograms:
                                                        plt.plot(histograms[color], label=color.upper())
                                                    plt.legend()
                                                    plt.grid(True)
                                                    plt.show()

                                                processor.collect_data(img_med, mask, largest_contour, chain_code, measures, histograms)
                                                processor.save_data_to_csv("extracted_medicine_features.csv")
                                            else:
                                                print("No valid measures found.")
                                        else:
                                            print("Chain code computation failed.")
                                    else:
                                        print("Contours could not be extracted.")
                                else:
                                    print("Mask creation failed.")
                            else:
                                print("Background removal failed.")
                        else:
                            print("Binary transform failed")
                    else:
                        print("Laplacian filtering failed.")
                else:
                    print("Image cropping failed.")
            else:
                print("Marked image creation failed.")
        else:
            print("Homogenization failed.")
    else:
        print("Stag detection failed.")

if __name__ == "__main__":
    main()

