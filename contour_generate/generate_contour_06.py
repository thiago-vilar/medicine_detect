import cv2
import numpy as np
import pickle
import csv
import os
from PIL import Image

def load_image(image_path):
    """Carrega uma imagem colorida."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagem não pode ser carregada. Verifique o caminho do arquivo.")
    return img

def convert_to_grayscale(image):
    """Converte a imagem para escala de cinza."""
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return img_gray

def convert_gray_to_binary(img_gray):
    """Converte a imagem de escala de cinza para binária."""
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def transform_alpha_mask(thresh):
    """Retira a máscara alpha da imagem"""
    png = thresh.convert('RGBA') 
    background = Image.new('RGBA', png.size, (255,255,255))
    alpha_composite_off = Image.alpha_composite(background, png)
    return alpha_composite_off [:, :, :3]

def draw_contour_on_binary_img(alpha_composite, image):
    """Desenha o contorno sobre a imagem binária"""
    contours, hierarchy = cv2.findContours(image=alpha_composite, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    image_copy = image.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    return image_copy

def save_image(image, directory="saved", base_filename="contour_detected"):
    """Salva a imagem em um diretório específico com numeração sequencial."""
    if not os.path.exists(directory):
        os.makedirs(directory)

    i = 0
    while os.path.exists(os.path.join(directory, f"{base_filename}{i}.png")):
        i += 1
    filename = os.path.join(directory, f"{base_filename}{i}.png")
    cv2.imwrite(filename, image)
    return filename

def main():
    image_path = input("Digite o caminho da imagem: ")
    image = load_image(image_path)
    img_gray = convert_to_grayscale(image)
    thresh = convert_gray_to_binary(img_gray)
    # alpha_composite = transform_alpha_mask(thresh)
    contour_image = draw_contour_on_binary_img(thresh, image)
    save_path = save_image(contour_image)
    print(f"Imagem com contornos salvos em: {save_path}")

if __name__ == "__main__":
    main()
