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

def create_mask(img, lower_bound, upper_bound):
    mask = cv2.inRange(img, lower_bound, upper_bound)
    return mask

def find_and_draw_contours(img, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(img)
    cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 3)
    return contour_img, contours

def normalize_contours(contours):
    """ Normaliza os contornos para a origem (0,0). """
    normalized_contours = []
    for cnt in contours:
        x_min, y_min = cnt.min(axis=0)[0]
        normalized_contour = cnt - [x_min, y_min]
        normalized_contours.append(normalized_contour.astype(np.int32))
    return normalized_contours

def save_contours(contours, directory="contorno"):
    """ Salva os contornos em formato PKL com numeração crescente. """
    if not os.path.exists(directory):
        os.makedirs(directory)
    index = 0
    filename = os.path.join(directory, f"contour_{index}.pkl")
    while os.path.exists(filename):
        index += 1
        filename = os.path.join(directory, f"contour_{index}.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(contours, f)
    print(f"Contours saved to {filename}")

def display_contours(img, contours):
    """ Exibe os contornos na imagem usando matplotlib. """
    for contour in contours:
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Normalized Contours")
    plt.axis('off')
    plt.show()

def main():
    filepath = input("Digite o caminho da imagem: ")
    img_no_bg = remove_background(filepath)
    lower = np.array([30, 30, 30])
    upper = np.array([250, 250, 250])
    mask = create_mask(img_no_bg, lower, upper)
    img_with_contours, contours = find_and_draw_contours(img_no_bg, mask)
    normalized_contours = normalize_contours(contours)
    display_contours(img_no_bg, normalized_contours)
    save_contours(normalized_contours)

if __name__ == "__main__":
    main()
