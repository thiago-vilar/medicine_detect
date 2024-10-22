import cv2
import numpy as np
import pickle
import os
from rembg import remove
import matplotlib.pyplot as plt

def remove_background(image_path):
    """ Remove o fundo da imagem usando rembg. """
    input_image = open(image_path, 'rb').read()
    output_image = remove(input_image)
    image_np_array = np.frombuffer(output_image, np.uint8)
    img = cv2.imdecode(image_np_array, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:  # Verifica se há canal alfa
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Remove o canal alfa
    return img

def convert_to_grayscale(img):
    """ Converte a imagem para escala de cinza. """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def binarize_image(gray_img):
    """ Binariza a imagem invertendo as cores do objeto e do fundo. """
    _, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    return thresh_img

def extract_contours(binarized_img):
    """ Extrai contornos da imagem binarizada e os normaliza. """
    contours, _ = cv2.findContours(binarized_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    normalized_contours = [cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True) for cnt in contours]
    return normalized_contours

def save_contours(contours, directory="contours"):
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
    """ Exibe os contornos na imagem usando matplotlib em rosa-quente. """
    img_display = img.copy()
    cv2.drawContours(img_display, contours, -1, (255, 0, 255), 3)  
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    plt.title("Extracted Contours")
    plt.show()

def main():
    image_path = input("Digite o caminho da imagem: ")
    img_no_bg = remove_background(image_path)
    gray_img = convert_to_grayscale(img_no_bg)
    bin_img = binarize_image(gray_img)
    contours = extract_contours(bin_img)
    display_contours(img_no_bg, contours)
    save_contours(contours)

if __name__ == "__main__":
    main()
