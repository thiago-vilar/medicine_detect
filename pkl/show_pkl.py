import cv2
import numpy as np
import pickle

def load_contour_data(filename):
    """Carrega dados de contornos de um arquivo .pkl."""
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def draw_and_show_contour(data):
    """Desenha e exibe contornos carregados de dados."""
    # Criar uma imagem em branco para desenhar os contornos
    # Supondo que os contornos estão em uma escala que cabe em uma imagem de 800x800
    image = np.zeros((300, 300, 3), dtype=np.uint8)

    # Descompactar dados
    contour_data = data['contour']
    # Convertendo de volta para um array NumPy
    contour = np.array(contour_data, dtype=np.int32)

    # Desenhar o contorno na imagem
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Cor verde com espessura 2

    # Exibir a imagem
    cv2.imshow("Contour", image)
    cv2.waitKey(0)  # Aguarda por uma tecla ser pressionada
    cv2.destroyAllWindows()

def main():
    filename = input("Digite o caminho para o arquivo de contorno (.pkl): ")
    contour_data = load_contour_data(filename)
    draw_and_show_contour(contour_data)

if __name__ == "__main__":
    main()
