import cv2
import numpy as np
import pickle
import os
from rembg import remove

def load_image_and_remove_bg(image_path):
    try:
        with open(image_path, 'rb') as file:
            input_image = file.read()
        output_image = remove(input_image)
        image_np_array = np.frombuffer(output_image, np.uint8)
        img = cv2.imdecode(image_np_array, cv2.IMREAD_UNCHANGED)
        return img[:, :, :3] if img is not None else None
    except Exception as e:
        print(f"Erro ao carregar ou processar a imagem: {e}")
        return None

def create_mask(img):
    lower_bound = np.array([30, 30, 30])
    upper_bound = np.array([250, 250, 250])
    return cv2.inRange(img, lower_bound, upper_bound)

def extract_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def normalize_contour(contour):
    min_x, min_y = np.min(contour, axis=0).reshape(-1)
    return contour - [min_x, min_y]

def calculate_hu_moments(contour):
    moments = cv2.moments(contour)
    huMoments = cv2.HuMoments(moments)
    return np.log(np.abs(huMoments))  # Log transform to handle very small values and make scale invariant

def draw_contours(img, contour):
    if contour is not None:
        return cv2.drawContours(img.copy(), [contour], -1, (0, 255, 0), 1)
    return img

def get_next_filename(directory, base_name="hu_moment"):
    files = [f for f in os.listdir(directory) if f.startswith(base_name) and f.endswith('.pkl')]
    max_number = 0
    for f in files:
        number_part = f.replace(base_name, "").replace(".pkl", "")
        if number_part.isdigit():
            max_number = max(max_number, int(number_part))
    return os.path.join(directory, f"{base_name}{max_number + 1}.pkl")

def save_hu_moments(huMoments, directory="hu_moments"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = get_next_filename(directory)
    with open(filename, 'wb') as f:
        pickle.dump(huMoments, f)
    print(f"Momentos de Hu salvos em {filename}")

def main():
    image_path = input("Digite o caminho da imagem: ")
    image = load_image_and_remove_bg(image_path)
    if image is None:
        print("Erro ao processar a imagem. Encerrando...")
        return
    mask = create_mask(image)
    contour = extract_contours(mask)
    if contour is None:
        print("Nenhum contorno significativo encontrado.")
        return
    contour = normalize_contour(contour)
    img_with_contours = draw_contours(image, contour)
    huMoments = calculate_hu_moments(contour)
    save_hu_moments(huMoments)

    cv2.imwrite('contoured_and_image_output.png', img_with_contours)
    print("Imagem com contornos e Momentos de Hu salvos: contoured_and_image_output.png")

if __name__ == "__main__":
    main()
