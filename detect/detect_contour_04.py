import cv2
import numpy as np
import pickle
import csv
import os

def load_and_display_image(image_path):
    """Carrega uma imagem colorida."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagem não pode ser carregada. Verifique o caminho do arquivo.")
    return img

def load_contour(contour_path):
    """Carrega um contorno de um arquivo .csv ou .pkl."""
    if contour_path.endswith('.pkl'):
        with open(contour_path, 'rb') as file:
            contour = pickle.load(file)
    elif contour_path.endswith('.csv'):
        with open(contour_path, newline='') as file:
            reader = csv.reader(file)
            contour = np.array(list(reader), dtype=np.int32)
    else:
        raise ValueError("Formato de arquivo de contorno não suportado. Use .csv ou .pkl.")
    return contour.reshape(-1, 1, 2)  # Reformata para o formato exigido pelo OpenCV

def draw_contour_on_image(image, contour):
    """Desenha o contorno especificado na imagem."""
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)  # Cor verde para o contorno
    return image

def main():
    image_path = input("Digite o caminho da imagem: ")
    contour_path = input("Digite o caminho do arquivo de contorno (.csv ou .pkl): ")

    image = load_and_display_image(image_path)
    contour = load_contour(contour_path)
    result_image = draw_contour_on_image(image, contour)

    cv2.imshow("Imagem com Contorno Especificado", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
