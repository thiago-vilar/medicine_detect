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
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
    return contour.reshape(-1, 1, 2)

def find_contours(gray_image):
    """Encontra contornos na imagem."""
    thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours


# def find_contours(gray_image):
#     """Encontra contornos na imagem."""
#     thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)[1]
#     contours, _ = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#     output_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    
#     cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)  

#     cv2.imshow('Contornos Encontrados', output_image)
#     cv2.waitKey(0)  
#     cv2.destroyAllWindows()
    
#     return contours

def match_shapes(contour1, contour2):
    """Compara dois contornos e retorna um valor de similaridade."""
    return cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)

def draw_contour_on_grayscale_image(gray_image, contour):
    """Desenha o contorno em verde sobre a imagem em escala de cinza."""
    color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(color_image, [contour], -1, (0, 255, 0), 3)
    return color_image

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
    contour_path = input("Digite o caminho do arquivo de contorno (.pkl): ")

    image = load_image(image_path)
    gray_image = convert_to_grayscale(image)
    loaded_contour = load_contour(contour_path)
    image_contours = find_contours(gray_image)

    for contour in image_contours:
        similarity = match_shapes(contour, loaded_contour)
        if similarity < 0.1:  # Ajuste este valor conforme necessário
            print(f"Contorno encontrado com alta similaridade: {similarity}")
            result_image = draw_contour_on_grayscale_image(gray_image, loaded_contour)
            saved_path = save_image(result_image)
            print(f"Imagem salva em: {saved_path}")
            cv2.imshow("Imagem com Contorno Especificado", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
    else:
        print("Nenhum contorno altamente similar foi encontrado.")

if __name__ == "__main__":
    main()
