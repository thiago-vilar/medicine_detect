import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from rembg import remove
import pickle
import os

def remove_background(filepath):
    """Remove o fundo da imagem especificada no caminho do arquivo."""
    input_image = open(filepath, 'rb').read()
    output_image = remove(input_image)
    image_np_array = np.frombuffer(output_image, np.uint8)
    img = cv2.imdecode(image_np_array, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Falha ao decodificar a imagem processada.")
    return img[:, :, :3]

def create_mask(img):
    """Cria uma máscara binária para a imagem com base em um intervalo de cores."""
    lower_bound = np.array([30, 30, 30])
    upper_bound = np.array([250, 250, 250])
    return cv2.inRange(img, lower_bound, upper_bound)

def find_centroid(mask):
    """Encontra o centróide da maior região branca na máscara."""
    M = cv2.moments(mask)
    if M["m00"] != 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None

def extract_and_draw_contours(img, mask):
    """Extrai contornos da máscara e desenha-os na imagem."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_with_contours = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)
    centroid = find_centroid(mask)
    if centroid:
        cv2.circle(img_with_contours, centroid, 5, (255, 0, 0), -1)
    return img_with_contours, contours

def crop_to_contours(img, contours):
    """Recorta a imagem para a bounding box mínima que envolve todos os contornos, com margem de 10%."""
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        margin_x = int(0.1 * w)
        margin_y = int(0.1 * h)
        x, y, w, h = x - margin_x, y - margin_y, w + 2 * margin_x, h + 2 * margin_y
        x, y = max(x, 0), max(y, 0)
        cropped_image = img[y:min(y+h, img.shape[0]), x:min(x+w, img.shape[1])]
        # Assegura fundo branco
        if cropped_image.shape[2] == 4:  # Se tem canal alpha
            return transform_alpha_mask_to_white_background(cropped_image)
        return cropped_image
    return img

def transform_alpha_mask_to_white_background(image):
    """Transforma uma imagem com canal alpha em uma imagem com fundo branco."""
    if image.shape[2] == 4:  # BGRA
        alpha_channel = image[:, :, 3]
        rgb_channels = image[:, :, :3]
        white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255
        alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
        foreground = alpha_factor * rgb_channels.astype(np.float32)
        background = (1.0 - alpha_factor) * white_background_image.astype(np.float32)
        composite_image = foreground + background
        return composite_image.astype(np.uint8)
    return image  # Assume already RGB

def save_largest_contour(contours, centroid):
    """Salva apenas o maior contorno fechado encontrado transladado para que o centróide esteja na origem."""
    if contours and centroid:
        largest_contour = max(contours, key=cv2.contourArea)
        largest_contour = largest_contour.reshape(-1, 2)  # Convertendo para (N, 2)
        translated_contour = largest_contour - np.array(centroid)
        # Assegura que o contorno seja fechado
        if not np.array_equal(translated_contour[0], translated_contour[-1]):
            translated_contour = np.vstack([translated_contour, translated_contour[0]])
        directory = 'contours'
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_number = len(os.listdir(directory)) + 1
        contour_path = os.path.join(directory, f"largest_cont_{file_number:03d}.pkl")
        with open(contour_path, 'wb') as f:
            pickle.dump(translated_contour, f)
        print(f"Contour saved with centroid at origin: {os.path.abspath(contour_path)}")

def save_image(img):
    """Salva a imagem recortada em formato PNG e imprime o caminho do arquivo."""
    directory = 'images'
    if not os.path.exists(directory):
        os.makedirs(directory)
    image_path = os.path.join(directory, f"output_image_{len(os.listdir(directory)) + 1:03d}.png")
    Image.fromarray(img).save(image_path)
    print(f"Image saved as: {os.path.abspath(image_path)}")

def main():
    filepath = input("Digite o caminho da imagem: ")
    img_no_bg = remove_background(filepath)
    mask = create_mask(img_no_bg)
    img_with_contours, contours = extract_and_draw_contours(img_no_bg, mask)
    cropped_img = crop_to_contours(img_with_contours, contours)
    centroid = find_centroid(mask)
    if centroid:
        save_largest_contour(contours, centroid)
    img_white_bg = transform_alpha_mask_to_white_background(cropped_img)
    save_image(img_white_bg)

    plt.imshow(img_white_bg)
    plt.title("Imagem com Contornos e Centróide")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
