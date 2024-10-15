import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_contours(filename):
    """ Carrega contornos de um arquivo .pkl. """
    with open(filename, 'rb') as f:
        contours = pickle.load(f)
    return contours

def get_canvas_size(contours):
    """ Determina o tamanho mínimo da tela necessária para desenhar todos os contornos. """
    if not contours:
        return (500, 500)  # Tamanho padrão se não houver contornos

    try:
        all_points = np.vstack([cnt for cnt in contours if cnt.shape[0] > 0])
        if all_points.size == 0:
            return (500, 500)  # Tamanho padrão se não houver pontos válidos
        x_max, y_max = np.max(all_points, axis=0)
        x_min, y_min = np.min(all_points, axis=0)
        width, height = x_max - x_min + 10, y_max - y_min + 10  # +10 para margem
        return (int(height), int(width))
    except ValueError as e:
        print("Error calculating canvas size:", e)
        return (500, 500)

def display_contours(contours):
    """ Exibe os contornos em uma janela. """
    height, width = get_canvas_size(contours)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title("Loaded Contours")
    plt.show()

def main():
    filename = input("Digite o caminho completo do arquivo .pkl: ")
    contours = load_contours(filename)
    if contours:
        display_contours(contours)
    else:
        print("No contours found in the file.")

if __name__ == "__main__":
    main()
