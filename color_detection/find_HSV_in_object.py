from rembg import remove
from PIL import Image
import os
from datetime import datetime
import cv2
import numpy as np

# Função para remover o fundo de uma imagem
def remove_background(input_path):
    if not os.path.exists(input_path):
        print(f"❌ Erro: A imagem {input_path} não foi encontrada.")
        return None

    input_image = Image.open(input_path)
    output_image = remove(input_image)

    output_dir = './frame'
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'img_{timestamp}.png')
    output_image.save(output_path)
    print(f"✔️ Imagem sem fundo salva em: {output_path}")
    return output_path

def find_dominant_threshold(image_path, hue_var_percent=5, sat_var_percent=43, val_var_percent=74):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Aplicar uma máscara para ignorar pixels de baixo valor
    mask = hsv_image[:, :, 2] > 30
    hsv_image = hsv_image[mask]

    # Calcular a moda dos canais HSV
    if hsv_image.size == 0:
        print("Nenhum pixel relevante encontrado após remoção do fundo.")
        return

    hist_hue = cv2.calcHist([hsv_image[:, 0]], [0], None, [180], [0, 180])
    hist_sat = cv2.calcHist([hsv_image[:, 1]], [0], None, [256], [0, 256])
    hist_val = cv2.calcHist([hsv_image[:, 2]], [0], None, [256], [0, 256])

    dominant_hue = np.argmax(hist_hue)
    dominant_sat = np.argmax(hist_sat)
    dominant_val = np.argmax(hist_val)

    # Aplicar os percentuais de variação para calcular os limites
    hue_var = int((hue_var_percent / 100) * dominant_hue)
    sat_var = int((sat_var_percent / 100) * dominant_sat)
    val_var = int((val_var_percent / 100) * dominant_val)

    low_threshold = [
        max(0, dominant_hue - hue_var),
        max(0, dominant_sat - sat_var),
        max(0, dominant_val - val_var)
    ]
    high_threshold = [
        min(179, dominant_hue + hue_var),
        min(255, dominant_sat + sat_var),
        min(255, dominant_val + val_var)
    ]

    print(f"Thresholds copied to clipboard: Low and High: {low_threshold}, {high_threshold}")

if __name__ == "__main__":
    input_path = input("Digite o caminho da imagem de entrada: ")
    output_path = remove_background(input_path)
    if output_path:
        # Parâmetros de variação podem ser ajustados aqui
        find_dominant_threshold(output_path, 9, 50, 35)
