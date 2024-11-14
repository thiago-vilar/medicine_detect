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
        self.pixel_size_mm = None  # Millimeters per pixel

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
            self.pixel_size_mm = 20.0 / width_px  # Assuming the stag is 20 mm wide

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
        # #Save
        # if not os.path.exists('features/cropped_imgs'):
        #     os.makedirs('features/cropped_imgs')
        # file_number = 0
        # while os.path.exists(f'features/cropped/img_cropped_{file_number}.jpg'):
        #     file_number += 1
        # cv2.imwrite(f'features/cropped/img_cropped_{file_number}.jpg', cropped_image)
        # print(f'Image saved as img_cropped_{file_number}.jpg')
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
        "Calculates and returns the grayscale histogram of an image, excluding any transparent pixels defined by the alpha channel."
        histogram = {}
        if img_med.shape[2] == 4:
            alpha_mask = img_med[:, :, 3] > 10
            img_bgr = img_med[alpha_mask, :3]
            img_gray = cv2.cvtColor(img_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY)

            # Calculate histogram for the grayscale image
            hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
            histogram['gray'] = hist.flatten()

            # Uncomment the following code to save the histogram to a file
            # directory = 'features/histogram'
            # if not os.path.exists(directory):
            #     os.makedirs(directory)
            # file_number = 0
            # file_path = os.path.join(directory, f'histogram_{file_number}.pkl')
            # while os.path.exists(file_path):
            #     file_number += 1
            #     file_path = os.path.join(directory, f'histogram_{file_number}.pkl')
            # with open(file_path, 'wb') as file:
            #     pickle.dump(histogram, file)
            # print(f"Histograms saved to {file_path}")
        return histogram
    
    def gaussian_filter(self, img_med):
        image_gray = cv2.cvtColor(img_med, cv2.COLOR_BGR2GRAY)
        img_gau_gray = cv2.GaussianBlur(image_gray, (35, 35), 0)
        return img_gau_gray
    
    def sobel_filter(img_gau_gray, scale=1, delta=0, ddepth=cv2.CV_16S):
        """
        Apply the Sobel filter to detect edges in both horizontal and vertical directions.
        
        Parameters:
        img_gau_gray : numpy.ndarray
            The input grayscale image.
        scale : int, optional
            The scale factor for the computed derivative values.
        delta : int, optional
            The delta value added to the results prior to storing them.
        ddepth : int, optional
            The desired depth of the destination image.
        
        Returns:
        numpy.ndarray
            The image resulting from the Sobel filter.
        """
        # Gradient X
        grad_x = cv2.Sobel(img_gau_gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta)
        abs_grad_x = cv2.convertScaleAbs(grad_x)

        # Gradient Y
        grad_y = cv2.Sobel(img_gau_gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        # Total Gradient (approximate)
        img_sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        return img_sobel
    
    def otsu(self, img_gau):
        _, otsu_binarized_image = cv2.threshold(img_gau, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu_binarized_image
    
    def create_mask(self, img):
        """Creates a binary mask for the foreground object in the image and saves it with transparency."""
        if img.shape[2] == 4:
            img = img[:, :, :3]  # Remove alpha channel
        lower_bound = np.array([30, 30, 30])
        upper_bound = np.array([255, 255, 255])
        mask = cv2.inRange(img, lower_bound, upper_bound)
        # Convert binary mask to 4-channel 
        mask_rgba = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGBA)
        mask_rgba[:, :, 3] = mask  # Set alpha channel to mask
        # # Save
        # directory = 'features/mask'
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # file_number = 0
        # while os.path.exists(f'{directory}/mask_{file_number}.png'):
        #     file_number += 1
        # cv2.imwrite(f'{directory}/mask_{file_number}.png', mask_rgba)
        # print(f'Mask saved as mask_{file_number}.png with transparency in {directory}')
        return mask
    
    def find_and_draw_contours(self, mask):
        """Finds and draws only the largest contour around the foreground object based on the mask and saves the image with alpha transparency."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            if largest_contour.size > 0:
                mask_with_contours = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
                mask_with_contours[:, :, 3] = mask  
                cv2.drawContours(mask_with_contours, [largest_contour], -1, (0, 0, 255, 255), 2)  
                # #Save             
                # directory = 'features/contour'
                # if not os.path.exists(directory):
                #     os.makedirs(directory)
                # file_number = 0
                # while os.path.exists(f'{directory}/contour_{file_number}.png'):
                #     file_number += 1
                # cv2.imwrite(f'{directory}/contour_{file_number}.png', mask_with_contours)
                # print(f'Contour image saved as contour_{file_number}.png in {directory}')
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
        # directory = 'features/signature'
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

    def draw_chain_code(self, img_med, contour, chain_code):
        ''' Draws the chain code on the image to visually represent contour direction changes. '''
        start_point = tuple(contour[0][0])
        current_point = start_point
        moves = {
            0: (1, 1),    # bottom-right
            1: (0, 1),    # right
            2: (-1, 1),   # top-right
            3: (-1, 0),   # left
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
        return img_med, len(chain_code)

    def medicine_measures(self, cropped_img, largest_contour):
        ''' Measures the dimensions of the detected contours and appends to a list. '''
        if not largest_contour:
            print("No contours found.")
            return None
        stag_width_px = np.max(self.corners[:, 0]) - np.min(self.corners[:, 0])
        px_to_mm_scale = 20 / stag_width_px
        measured_img = cropped_img.copy()
        for point in largest_contour:
            x, y, w, h = cv2.boundingRect(point)
            width_mm = w * px_to_mm_scale
            height_mm = h * px_to_mm_scale
            cv2.rectangle(measured_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(measured_img, f"{width_mm:.1f}mm x {height_mm:.1f}mm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # # Save
        # directory = 'features/medicine_measures'
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # file_number = 0
        # file_path = os.path.join(directory, f'med_measure_{file_number}.png')
        # while os.path.exists(file_path):
        #     file_number += 1
        #     file_path = os.path.join(directory, f'med_measure_{file_number}.png')
        # cv2.imwrite(file_path, measured_img)
        # print(f"Image saved as {file_path}")

        return measured_img

if __name__ == "__main__":
    # image_path = ".\\frames\\new_test_light\\thiago_fotos_10_down_lighton_ampoules\\color_c1.jpg"
    image_path = ".\\frames\\matte black box\\img_2_007.jpg"
    
    stag_id = 2
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
                        plt.title('Background Removed')
                        plt.show()

                        # Supondo que img_med é sua imagem carregada e processada
                        histogram = processor.calculate_histograms(img_med)

                        # Plot do histograma em escala de cinza
                        plt.figure(figsize=(10, 5))
                        plt.title('Histograma em Escala de Cinza')
                        plt.xlabel('Intensidade do Pixel')
                        plt.ylabel('Quantidade de Pixels')
                        plt.plot(histogram['gray'], color='gray', label='GRAY')
                        plt.xlim([0, 256])
                        plt.legend()
                        plt.grid(True)
                        plt.show()

                        gau_img = processor.gaussian_filter(img_med)
                        plt.imshow(gau_img, cmap='gray')
                        plt.title ('Gaussian Image')
                        plt.show()

                        sobel_img = processor.sobel_filter(gau_img)
                        plt.imshow(sobel_img, cmap= 'gray')
                        plt.title ('Sobel Image')    

                        otsu_img = processor.otsu(gau_img)
                        plt.imshow(otsu_img, cmap='gray')
                        plt.title ('Otsu Image')
                        plt.show()

                        mask = processor.create_mask(background_removed)
                        plt.imshow(mask, cmap='gray')
                        plt.title('Mask')
                        plt.show()

                        contoured_image, largest_contour = processor.find_and_draw_contours(mask)
                        if contoured_image is not None and largest_contour is not None and largest_contour.size > 0:
                            plt.imshow(cv2.cvtColor(contoured_image, cv2.COLOR_BGR2RGB))
                            plt.title('Largest Contour by Mask')
                            plt.show()

                            chain_code, _ = processor.compute_chain_code(largest_contour)  
                            chain_drawn_image, _ = processor.draw_chain_code(img_med, largest_contour, chain_code)
                            plt.imshow(cv2.cvtColor(chain_drawn_image, cv2.COLOR_BGR2RGB))
                            plt.title('Chain Code Drawn')
                            plt.show()

                            measured_medicine = processor.medicine_measures(img_med, [largest_contour])
                            plt.imshow(cv2.cvtColor(measured_medicine, cv2.COLOR_BGR2RGB))
                            plt.title('Measured Medicine')
                            plt.show()


    else:
        print("Stag detection failed.")


