import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import os
import pickle

def extract_texture_features(image_np_array):
    '''Extrai características de textura usando Local Binary Patterns.'''
    # Parâmetros para LBP
    radius = 3
    n_points = 8 * radius
    method = 'uniform'
    
    lbp_image = local_binary_pattern(image_np_array, n_points, radius, method)
    
    # Normaliza o histograma
    n_bins = int(lbp_image.max() + 1)
    hist, _ = np.histogram(lbp_image, density=True, bins=n_bins, range=(0, n_bins))
    
    return hist

def display_texture_features(hist):
    '''Exibe o histograma de características de textura.'''
    plt.figure(figsize=(8, 4))
    plt.title("Histograma de Características de Textura LBP")
    plt.xlabel("Bins")
    plt.ylabel("Frequência")
    plt.bar(range(len(hist)), hist, width=1.0)
    plt.show()

def main():
    filepath = input("Digite o caminho da imagem: ")
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Erro ao carregar a imagem.")
        return

    # Extrai características de textura
    texture_features = extract_texture_features(image)
    
    # Exibe as características de textura
    display_texture_features(texture_features)

if __name__ == "__main__":
    main()
