import cv2
import numpy as np
import matplotlib.pyplot as plt
import stag
from rembg import remove
import datetime
import pickle
import os


def detect_stag_markers(image_path, library_hd=17, error_correction=None):
    """
    Detecta marcadores STag em uma imagem utilizando a biblioteca stag-python.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Erro ao carregar a imagem.")
        return None, None

    config = {'libraryHD': library_hd}
    if error_correction is not None:
        config['errorCorrection'] = error_correction

    corners, ids, rejected = stag.detectMarkers(image, **config)
    return corners, ids, image

def crop_image_around_marker(image, corners, ids, marker_id, scaling_factor=4):
    for i, id_ in enumerate(ids):
        if id_[0] == marker_id:
            corner = corners[i]
            x_min, y_min = np.min(corner, axis=0)
            x_max, y_max = np.max(corner, axis=0)
            width = x_max - x_min
            height = y_max - y_min
            
            # Calcula o novo tamanho baseado no fator de escala
            new_width = width * scaling_factor
            new_height = height * scaling_factor

            # Calcula o novo centro para o corte
            new_x_min = max(int(x_min - (new_width - width) / 2), 0)
            new_y_min = max(int(y_min - (new_height - height) / 2), 0)
            new_x_max = min(int(x_max + (new_width - width) / 2), image.shape[1])
            new_y_max = min(int(y_max + (new_height - height) / 2), image.shape[0])

            cropped_image = image[new_y_min:new_y_max, new_x_min:new_x_max]
            return cropped_image
    return None


def main():
    image_path = input("Por favor, insira o caminho da imagem: ")
    corners, ids, image = detect_stag_markers(image_path)

    if ids is not None and len(ids) > 0:
        print(f"Marcadores STag detectados com IDs: {ids[:,0]}")
        marker_id = int(input("Digite o ID do marcador que deseja recortar: "))
        cropped_image = crop_image_around_marker(image, corners, marker_id)

        if cropped_image is not None:
            cv2.imshow('Imagem Recortada', cropped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Marcador não encontrado ou erro ao recortar.")
    else:
        print("Nenhum marcador STag foi encontrado.")

if __name__ == "__main__":
    main()
