import cv2
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
import datetime
import pickle
import os

def remove_background(filepath):
    input_image = open(filepath, 'rb').read()
    output_image = remove(input_image)
    image_np_array = np.frombuffer(output_image, np.uint8)
    img = cv2.imdecode(image_np_array, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Falha ao decodificar a imagem processada.")
    return img[:, :, :3]  # Remove o canal alpha se houver

def read_and_convert_image(image_array):
    img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return img

def create_mask(img, lower_bound, upper_bound):
    mask = cv2.inRange(img, lower_bound, upper_bound)
    return mask

def find_and_draw_contours(img, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_with_contours = cv2.drawContours(img.copy(), contours, -1, (255, 0, 0), 3)
    return img_with_contours, contours

def draw_bounding_box(img, contour):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
    return img

def crop_and_draw_upper_half_contour(img, contour):
    x, y, w, h = cv2.boundingRect(contour)
    cropped_height = h // 2
    cropped_img = img[y:y+cropped_height, x:x+w]

    contour_points = contour[:, 0, :]
    y_min = np.min(contour_points[:, 1])
    y_max = np.max(contour_points[:, 1])
    y_threshold = y_min + (y_max - y_min) / 2

    upper_half_indices = np.where(contour_points[:, 1] <= y_threshold)[0]
    upper_half_contour = contour_points[upper_half_indices]
    contour_shifted = upper_half_contour - np.array([x, y])

    # Utilizando matplotlib para desenhar os pontos sem conectar
    fig, ax = plt.subplots()
    ax.imshow(cropped_img)
    ax.scatter(contour_shifted[:, 0], contour_shifted[:, 1], color='green', s=1)
    ax.set_axis_off()
    plt.show()

    return cropped_img, contour_shifted

def save_contour_signature(contour_shifted):
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f'signature_contours/{current_time}.pkl'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(contour_shifted, f)

def main():
    filepath = input("Digite o caminho da imagem: ")
    img_no_bg = remove_background(filepath)
    img = read_and_convert_image(img_no_bg)

    lower = np.array([30, 30, 30])
    upper = np.array([250, 250, 250])
    mask = create_mask(img, lower, upper)

    img_with_contours, contours = find_and_draw_contours(img, mask)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        img_with_box = draw_bounding_box(img_with_contours, largest_contour)
        cropped_img, contour_shifted = crop_and_draw_upper_half_contour(img, largest_contour)
        save_contour_signature(contour_shifted)
    else:
        print("Nenhum contorno encontrado.")

if __name__ == "__main__":
    main()
