import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rembg import remove

def remove_background(image_np_array):
        """Removes the background from the image using the rembg library."""
        is_success, buffer = cv2.imencode(".png", image_np_array)
        if not is_success:
            raise ValueError("Failed to encode image for background removal.")
        output_image = remove(buffer.tobytes())
        img_med = cv2.imdecode(np.frombuffer(output_image, np.uint8), cv2.IMREAD_UNCHANGED)
        if img_med is None:
            raise ValueError("Failed to decode processed image.")
        return img_med    

def box_blur(image):
        filter = np.array([
            [ 0,  1, 0],
            [ 0, 1, 0],
            [ 0,  1, 0]

        ])
        
        ddepth = cv2.CV_16S
        img_filtered = cv2.filter2D(image, ddepth, filter)
        abs_img_filtered = cv2.convertScaleAbs(img_filtered)
        return abs_img_filtered

def laplacian(image):
        """Applies Gaussian blur followed by a Sobel filter to enhance horizontal edges of a reflective object."""
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        filter = np.array([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ])
        ddepth = cv2.CV_16S
        img_filtered = cv2.filter2D(img_blur, ddepth, filter)
        abs_img_filtered = cv2.convertScaleAbs(img_filtered)
        return abs_img_filtered

def sobel_x(image):
        filter_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 2]
        ])

        ddepth = cv2.CV_16S
        img_filtered_x = cv2.filter2D(image, ddepth, filter_x)

        abs_img_filtered_x = cv2.convertScaleAbs(img_filtered_x)

        return abs_img_filtered_x

def sobel_y(image):
        filter_y = np.array([
            [-1, 2, 1],
            [ 0, 0, 0],
            [ 1, 2, 1]
        ])
        ddepth = cv2.CV_16S
        img_filtered_y = cv2.filter2D(image, ddepth, filter_y)
        abs_img_filtered_y = cv2.convertScaleAbs(img_filtered_y)
        return abs_img_filtered_y    



def prewitt_x(image):
        filter_x = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ])
    
        ddepth = cv2.CV_16S
        img_filtered_x = cv2.filter2D(image, ddepth, filter_x)
        abs_img_filtered_x = cv2.convertScaleAbs(img_filtered_x)
        return abs_img_filtered_x
        
def prewitt_y(image):

        filter_y = np.array([
            [-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]
        ])
        ddepth = cv2.CV_16S
        img_filtered_y = cv2.filter2D(image, ddepth, filter_y)
        abs_img_filtered_y = cv2.convertScaleAbs(img_filtered_y)
        return abs_img_filtered_y     


def edge_detection_1(image):
        filter = np.array([
            [ 1, 0,-1],
            [ 0, 0, 0],
            [-1, 0, 1]

        ])
        ddepth = cv2.CV_16S
        img_filtered = cv2.filter2D(image, ddepth, filter)
        abs_img_filtered = cv2.convertScaleAbs(img_filtered)
        return abs_img_filtered 

def edge_detection_2(image):
        filter = np.array([
            [  0, -1,  0],
            [ -1,  4, -1],
            [ 0,  -1,  0]
        ])
        ddepth = cv2.CV_16S
        img_filtered = cv2.filter2D(image, ddepth, filter)
        abs_img_filtered = cv2.convertScaleAbs(img_filtered)
        return abs_img_filtered 

def edge_detection_3(image):
        filter = np.array([
            [ -1, -1, -1],
            [ -1,  0, -1],
            [ -1, -1, -1]
        ])
        ddepth = cv2.CV_16S
        img_filtered = cv2.filter2D(image, ddepth, filter)
        abs_img_filtered = cv2.convertScaleAbs(img_filtered)
        return abs_img_filtered 

      
def gaussian_blur(image):
        filter = np.array([
            [ 1,  2, 1],
            [ 2, -4, 2],
            [ 1,  2, 1]
        ])
        ddepth = cv2.CV_16S
        img_filtered = cv2.filter2D(image, ddepth, filter)
        abs_img_filtered = cv2.convertScaleAbs(img_filtered)
        return abs_img_filtered 


def sharpen_1 (image):
        filter = np.array([
            [0,  1, 0],
            [1, -6, 1],
            [0,  1, 0]
        ])
        ddepth = cv2.CV_16S
        img_filtered = cv2.filter2D(image, ddepth, filter)
        abs_img_filtered = cv2.convertScaleAbs(img_filtered)
        return abs_img_filtered 

def sharpen_2 (image):
        filter = np.array([
            [0,  2, 0],
            [2, -9, 2],
            [0,  2, 0]
        ])
        ddepth = cv2.CV_16S
        img_filtered = cv2.filter2D(image, ddepth, filter)
        abs_img_filtered = cv2.convertScaleAbs(img_filtered)
        return abs_img_filtered 

def mexican (image):
        filter = np.array([
            [0,  0, -1, 0, 0],
            [0, -1, -2,-1, 0],
            [-1,-2, 16,-2,-1],
            [0, -1, -2,-1, 0],
            [0,  0, -1, 0, 0]
        ])
        ddepth = cv2.CV_16S
        img_filtered = cv2.filter2D(image, ddepth, filter)
        abs_img_filtered = cv2.convertScaleAbs(img_filtered)
        return abs_img_filtered 
      
def sepia (image):
        filter = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        ddepth = cv2.CV_16S
        img_filtered = cv2.filter2D(image, ddepth, filter)
        abs_img_filtered = cv2.convertScaleAbs(img_filtered)
        return abs_img_filtered 

def embossed_edges(image):
        filter = np.array([ 
            [-2, -1, 0],
            [-1,  5, 1],
            [0,  1,  2]
        ])
       
        ddepth = cv2.CV_16S
        img_filtered = cv2.filter2D(image, ddepth, filter)
        abs_img_filtered = cv2.convertScaleAbs(img_filtered)
        return abs_img_filtered 

def generate_gaussian_kernel(size, sigma):
    """Generate a Gaussian kernel using numpy"""
    kernel = np.zeros((size, size), dtype=np.float32)
    k = size // 2
    
    for i in range(-k, k+1):
        for j in range(-k, k+1):
            kernel[i+k, j+k] = np.exp(-(i**2 + j**2) / (2 * sigma**2))
    
    kernel_sum = np.sum(kernel)
    kernel = kernel / kernel_sum
    return kernel

def difference_of_gaussians(img):
    """Apply Difference of Gaussians using custom Gaussian kernels"""
    kernel1 = generate_gaussian_kernel(3, 1.0)
    kernel2 = generate_gaussian_kernel(5, 1.4)
    
    gauss1 = cv2.filter2D(img, -1, kernel1)
    gauss2 = cv2.filter2D(img, -1, kernel2)
    
    dog = cv2.subtract(gauss1, gauss2)
    return dog

def laplacian_of_gaussians(image):
    """Apply Laplacian of Gaussians using a custom Gaussian kernel followed by a Laplacian filter"""
    kernel_gauss = generate_gaussian_kernel(5, 1.4)
    smoothed = cv2.filter2D(image, -1, kernel_gauss)
    
    laplacian_kernel = np.array([[0, 1, 0], 
                                 [1, -5, 1], 
                                 [0, 1, 0]], dtype=np.float32)
    
    log = cv2.filter2D(smoothed, -1, laplacian_kernel)
    return log


def apply_blend_filter(image):
        """Transforms image to grayscale, applies Sobel filter, and overlays it on the original image."""
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        filter = np.array([
            [0,  1, 0],
            [1, -5, 1],
            [0,  1, 0]
        ])
        ddepth = cv2.CV_16S
        img_filtered = cv2.filter2D(img_blur, ddepth, filter)
        abs_img_filtered = cv2.convertScaleAbs(img_filtered)
        alpha = 0.7 
        blended = cv2.addWeighted(image, 1, cv2.cvtColor(abs_img_filtered, cv2.COLOR_GRAY2BGR), alpha, 0)
        return blended

def apply_subtraction_filter(image):
        """Converts image to grayscale, applies Laplacian filter, and subtracts it from the original image."""
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        filter = np.array([
            [0,  1, 0],
            [1, -3, 1],
            [0,  1, 0]
        ])
        ddepth = cv2.CV_16S
        img_filtered = cv2.filter2D(img_blur, ddepth, filter)
        abs_img_filtered = cv2.convertScaleAbs(img_filtered)
        abs_img_filtered_colored = cv2.cvtColor(abs_img_filtered, cv2.COLOR_GRAY2BGR)

        # Subtract the filtered image from the original image
        subtracted = cv2.subtract(image, abs_img_filtered_colored)
        return subtracted

def create_mask(img):
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

def add_filters(image):
    filter_x = sobel_x(image)
    filter_y = sobel_y(image)
    
    # Normalizando para evitar problemas de overflow
    img_sobelx = cv2.convertScaleAbs(filter_x )
    img_sobely = cv2.convertScaleAbs(filter_y)

    # Adição das imagens
    added_image = cv2.add(img_sobelx, img_sobely)
    return added_image

def subtract_filters(image):
    img_sobelx = prewitt_x(image)
    img_sobely = sobel_y(image)

    # Normalizando
    img_sobelx = cv2.convertScaleAbs(img_sobelx)
    img_sobely = cv2.convertScaleAbs(img_sobely)

    # Subtração das imagens
    subtracted_image = cv2.subtract(img_sobelx, img_sobely)
    return subtracted_image


filter_functions = [subtract_filters, box_blur, laplacian, sobel_x, sobel_y,  prewitt_x, prewitt_y, edge_detection_1, edge_detection_2, edge_detection_3, gaussian_blur, sharpen_1, sharpen_2, mexican, sepia, embossed_edges, laplacian_of_gaussians, apply_blend_filter, apply_subtraction_filter ]

if __name__ == '__main__':
    image_path = '.\\features\\cropped_imgs\\img_cropped_7.png'
    original_image = cv2.imread(image_path)

    if original_image is None:
        print("Image not found.")
    else:

        for filter_func in filter_functions:
            # Apply filter
            filtered_image = filter_func(original_image)
            
            # Remove background
            background_removed_image = remove_background(filtered_image)

            # Mask
            mask = create_mask(background_removed_image)
            
            # Show images
            plt.figure(figsize=(15, 5))
            plt.subplot(141), plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)), plt.title('Original')
            plt.subplot(142), plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)), plt.title(f'After Filter{filter_func.__name__}')
            plt.subplot(143), plt.imshow(cv2.cvtColor(background_removed_image, cv2.COLOR_BGR2RGB)), plt.title('Background Removed')
            plt.subplot(144), plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)), plt.title('Mask')
            plt.show()