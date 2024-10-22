import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_contour(contour_path):
    with open(contour_path, 'rb') as f:
        contour = pickle.load(f)
    return contour.reshape(-1, 1, 2)

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def sliding_window_search(image, contour, step_size=20, window_size=(100,100)):
    # Converte a imagem para escala de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Prepara uma imagem branca para desenhar os resultados
    result_image = np.dstack([gray_image]*3)  # Converte de volta para BGR para desenhar em colorido
    
    max_similarity = float('inf')
    best_position = (0, 0)

    for y in range(0, gray_image.shape[0] - window_size[1], step_size):
        for x in range(0, gray_image.shape[1] - window_size[0], step_size):
            # Cria uma máscara com o contorno na posição atual
            mask = np.zeros_like(gray_image)
            move_contour = contour + np.array([x, y])
            cv2.drawContours(mask, [move_contour], -1, 255, thickness=cv2.FILLED)

            # Recorta a região de interesse da imagem e da máscara
            roi_image = gray_image[y:y+window_size[1], x:x+window_size[0]]
            roi_mask = mask[y:y+window_size[1], x:x+window_size[0]]

            # Calcula a correlação entre a janela e o contorno
            correlation = cv2.matchShapes(roi_image, roi_mask, cv2.CONTOURS_MATCH_I1, 0.0)

            if correlation < max_similarity:
                max_similarity = correlation
                best_position = (x, y)

            # Desenha o contorno na posição testada
            cv2.drawContours(result_image, [move_contour], -1, (0, 255, 0), 3)

    # Desenha o melhor contorno encontrado
    best_contour = contour + np.array([best_position[0], best_position[1]])
    cv2.drawContours(result_image, [best_contour], -1, (255, 0, 0), 3)
    return result_image, best_position, max_similarity


def main():
    image_path = r'C:\Users\thiag\OneDrive\Documentos\dev\medicine_detect\frames\frame_20240924_110521.jpg'
    contour_path = r'C:\Users\thiag\OneDrive\Documentos\dev\medicine_detect\largest_contour.pkl'
    
    image = load_image(image_path)
    contour = load_contour(contour_path)
    
    result_image, best_position, max_similarity = sliding_window_search(image, contour, step_size=20, window_size=(100,100))
    
    print(f"Best position: {best_position} with similarity: {max_similarity}")
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title("Result Image with Sliding Window Search")
    plt.show()

if __name__ == "__main__":
    main()
