import cv2
import numpy as np
import json

def load_contour_from_json(filename):
    with open(filename, 'r') as file:
        contour_list = json.load(file)
    return np.array(contour_list, dtype=np.int32).reshape((-1, 1, 2))

def find_contour_in_image(image_path, contour_template):
    # Carregar imagem
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar a imagem de {image_path}.")
        return

    # Converter para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarizar a imagem
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Encontrar contornos na imagem binarizada
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tentar encontrar o contorno template em todos os contornos detectados na imagem
    found = False
    for cnt in contours:
        ret = cv2.matchShapes(contour_template, cnt, 1, 0.0)
        if ret < 0.05:  # Um valor baixo indica uma boa correspondência
            found = True
            # Desenhar o contorno correspondente na imagem em escala de cinza
            cv2.drawContours(gray, [cnt], -1, (255, 0, 0), 3)  # Cor azul em BGR
            print(f"Correspondência encontrada com valor: {ret}")
            break
    
    if not found:
        print("Nenhum contorno correspondente encontrado.")

    # Salvar a imagem resultante para garantir que o contorno seja visualizado
    result_image_path = 'resultado_contorno.jpg'
    cv2.imwrite(result_image_path, gray)
    print(f"Resultado salvo em: {result_image_path}")

    # Mostrar a imagem
    cv2.imshow("Resultado da Busca", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image_path = '.\\frames\\frame_20240924_110521.jpg'
    contour_path = '.\\contorno\\contour0.json'
    
    # Carregar o contorno do arquivo JSON
    contour_template = load_contour_from_json(contour_path)
    
    # Realizar a busca pelo contorno na imagem
    find_contour_in_image(image_path, contour_template)

if __name__ == "__main__":
    main()
