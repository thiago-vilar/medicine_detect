import cv2
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
from io import BytesIO

def remove_background(filepath):
    # Remove o fundo da imagem usando a função remove do rembg
    input_image = open(filepath, 'rb').read()
    output_image = remove(input_image)
    # Converte a imagem de saída para um formato utilizável pelo OpenCV
    image_np_array = np.frombuffer(output_image, np.uint8)
    img = cv2.imdecode(image_np_array, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Falha ao decodificar a imagem processada.")
    return img[:, :, :3]  # Remove o canal alpha se houver

def read_and_convert_image(image_array):
    # Converte para RGB
    img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return img

def create_mask(img, lower_bound, upper_bound):
    # Cria uma máscara com base em um intervalo de cores
    mask = cv2.inRange(img, lower_bound, upper_bound)
    return mask

def find_and_draw_contours(img, mask):
    # Encontra contornos e desenha na imagem original
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_with_contours = cv2.drawContours(img.copy(), contours, -1, (255, 0, 0), 3)
    return img_with_contours, contours

def draw_bounding_box(img, contour):
    # Desenha uma caixa delimitadora ao redor do maior contorno
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
    return img

def main():
    filepath = input("Digite o caminho da imagem: ")
    img_no_bg = remove_background(filepath)
    img = read_and_convert_image(img_no_bg)

    plt.figure()
    plt.imshow(img)
    plt.title("Imagem Original sem Fundo")

    lower = np.array([30, 30, 30])
    upper = np.array([250, 250, 250])
    mask = create_mask(img, lower, upper)

    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title("Máscara da Imagem")

    img_with_contours, contours = find_and_draw_contours(img, mask)
    
    plt.figure()
    plt.imshow(img_with_contours)
    plt.title("Imagem com Contornos")

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        img_with_box = draw_bounding_box(img, largest_contour)
        
        plt.figure()
        plt.imshow(img_with_box)
        plt.title("Imagem com Caixa Delimitadora")

    plt.show()

if __name__ == "__main__":
    main()
