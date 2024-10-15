import cv2
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
import pickle
import os
import csv
from PIL import Image, ImageDraw

def remove_background(filepath):
    input_image = open(filepath, 'rb').read()
    output_image = remove(input_image)
    image_np_array = np.frombuffer(output_image, np.uint8)
    img = cv2.imdecode(image_np_array, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Falha ao decodificar a imagem processada.")
    return img[:, :, :3]  # Remove o canal alpha se houver

def read_and_convert_image(image_array):
    img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return img

def create_mask(img, lower_bound, upper_bound):
    mask = cv2.inRange(img, lower_bound, upper_bound)
    return mask

def find_and_draw_contours(img, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_contours = cv2.drawContours(img.copy(), contours, -1, (255, 0, 0), 3)
    return img_with_contours, contours

def save_contours(contours):
    directory = "contorno"
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, contour in enumerate(contours):
        reshaped_contour = contour.reshape(-1, 2)
        
        # Save as Pickle
        filename_pkl = os.path.join(directory, f"contour{i}.pkl")
        with open(filename_pkl, 'wb') as f:
            pickle.dump(reshaped_contour, f)
        
        # Save as CSV
        filename_csv = os.path.join(directory, f"contour{i}.csv")
        with open(filename_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(reshaped_contour)
        
        # Save as PNG using Pillow
        image = Image.new('RGB', (500, 500), 'white')
        draw = ImageDraw.Draw(image)
        draw.polygon([tuple(p) for p in reshaped_contour], outline='black')
        filename_png = os.path.join(directory, f"contour{i}.png")
        image.save(filename_png)

        print(f"Contour {i} saved as Pickle, CSV, and PNG in {directory}/")

def main():
    filepath = input("Digite o caminho da imagem: ")
    img_no_bg = remove_background(filepath)
    img = read_and_convert_image(img_no_bg)

    lower = np.array([30, 30, 30])
    upper = np.array([250, 250, 250])
    mask = create_mask(img, lower, upper)

    img_with_contours, contours = find_and_draw_contours(img, mask)
    
    if contours:
        save_contours(contours)  

    plt.imshow(img_with_contours)
    plt.title("Imagem com Contornos")
    plt.axis('off')  
    plt.show()

if __name__ == "__main__":
    main()
