import cv2
import numpy as np
import pickle
import csv
import os

def load_image(image_path):
    """Carrega uma imagem colorida."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagem não pode ser carregada. Verifique o caminho do arquivo.")
    return img

def convert_to_grayscale(image):
    """Converte a imagem para escala de cinza."""
    return cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

def convert_gray_to_binary(image):
    """Converte a imagem de escala de cinza para Binária."""
  
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    # visualize the binary image
    cv2.imshow('Binary image', thresh)
    cv2.waitKey(0)
    cv2.imwrite('image_thres1.jpg', thresh)
    cv2.destroyAllWindows()


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
    contour_path = input("Digite o caminho do arquivo de contorno (.csv ou .pkl): ")

    image = load_image(image_path)
    gray_image = convert_to_grayscale(image)
    contour = load_contour(contour_path)
    result_image = draw_contour_on_grayscale_image(gray_image, contour)

    saved_path = save_image(result_image)
    print(f"Imagem salva em: {saved_path}")

    cv2.imshow("Imagem com Contorno Especificado", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
