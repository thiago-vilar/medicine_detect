import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

def load_contours_from_json(filename):
    """Carrega contornos de um arquivo JSON."""
    with open(filename, 'r') as file:
        contours = json.load(file)
    return [np.array(contour, dtype=np.int32) for contour in contours]

def display_contours(contours):
    """Exibe contornos em uma janela usando Matplotlib."""
    canvas = np.zeros((500, 500, 3), dtype=np.uint8)
    for contour in contours:
        cv2.drawContours(canvas, [contour], -1, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title('Contours')
    plt.show()

def main():
    filename = '.\\contorno\\contour0.json'  
    contours = load_contours_from_json(filename)
    display_contours(contours)

if __name__ == "__main__":
    main()

