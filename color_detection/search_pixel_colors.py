import cv2
import numpy as np

def detect_color_range_and_label(image_path, lower_bounds, upper_bounds):
    """
    Detecta cores em uma imagem dentro dos limites HSV especificados e rotula os pixels correspondentes.

    Parâmetros:
    - image_path: Caminho para a imagem que será processada.
    - lower_bounds: Lista com os limites inferiores [Hue, Saturação, Valor].
    - upper_bounds: Lista com os limites superiores [Hue, Saturação, Valor].
    """
    # Carregar a imagem
    image = cv2.imread(image_path)
    if image is None:
        print("Erro: Não foi possível abrir a imagem. Verifique se o caminho está correto.")
        return

    # Converter a imagem de BGR para HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Criar a máscara usando os limites especificados
    lower_bounds_array = np.array(lower_bounds, dtype=np.uint8)
    upper_bounds_array = np.array(upper_bounds, dtype=np.uint8)
    mask = cv2.inRange(hsv_image, lower_bounds_array, upper_bounds_array)

    # Calcular a taxa de pixels que correspondem à máscara
    total_pixels = mask.size
    matching_pixels = np.count_nonzero(mask)
    similarity_rate = (matching_pixels / total_pixels) * 100

    # Rotular pixels encontrados em azul no espaço de cor BGR
    labeled_image = image.copy()
    labeled_image[mask != 0] = [255, 0, 0]  # BGR para Azul

    # Salvar a imagem de resultado
    output_image_path = "labeled_image.jpg"
    cv2.imwrite(output_image_path, labeled_image)
    print(f"Imagem rotulada salva como '{output_image_path}'.")
    print(f"Taxa de similaridade: {similarity_rate:.2f}% - {matching_pixels} pixels correspondentes de {total_pixels} pixels totais.")

    # Opcional: Mostrar a imagem resultante
    cv2.imshow('Imagem Rotulada', labeled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

lower_bounds = [15, 86, 162]
upper_bounds = [17, 255, 255]

image_path = input("Digite o caminho completo ou relativo da imagem: ")

detect_color_range_and_label(image_path, lower_bounds, upper_bounds)
