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

def crop_and_extract_upper_half_contour(img, contour):
    x, y, w, h = cv2.boundingRect(contour)
    cropped_height = h // 2
    contour_points = np.vstack([pt for pt in contour if pt[0][1] <= y + cropped_height])
    contour_shifted = contour_points - np.array([x, y])
    blank_image = np.zeros_like(img)
    cv2.polylines(blank_image, [contour_shifted], isClosed=False, color=(0, 255, 0), thickness=2)
    return blank_image, contour_shifted

def save_contour_signature(contour_shifted):
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f'pkl/img{current_time}.pkl'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        np_contour = np.array(contour_shifted, dtype=np.int32)
        pickle.dump(np_contour, f)

def main():
    filepath = input("Digite o caminho da imagem: ")
    img_no_bg = remove_background(filepath)
    img = read_and_convert_image(img_no_bg)

    lower = np.array([30, 30, 30])
    upper = np.array([250, 250, 250])
    mask = create_mask(img, lower, upper)
    img_with_contours = find_and_draw_contours(img, mask)

    if img_with_contours[1]:
        largest_contour = max(img_with_contours[1], key=cv2.contourArea)
        img_with_upper_contour, contour_shifted = crop_and_extract_upper_half_contour(img, largest_contour)
        save_contour_signature(contour_shifted)
        print("Contorno superior salvo com sucesso.")
    else:
        print("Nenhum contorno encontrado.")

if __name__ == "__main__":
    main()
