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
        if img is None:
            return None
        # Certifique-se de que a imagem tem 3 canais
        if img.shape[2] == 4:  # Se tiver canal alfa
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.shape[2] == 1:  # Se for escala de cinza
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
    except Exception as e:
        print(f"Erro ao carregar ou processar a imagem: {e}")
        return None

def create_mask(img):
    lower_bound = np.array([30, 30, 30])
    upper_bound = np.array([250, 250, 250])
    return cv2.inRange(img, lower_bound, upper_bound)

def extract_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def calculate_hu_moments(contour):
    moments = cv2.moments(contour)
    huMoments = cv2.HuMoments(moments)
    return np.log(np.abs(huMoments) + 1e-10) 

def load_hu_moments(filename):
    with open(filename, 'rb') as file:
        huMoments = pickle.load(file)
    return huMoments

def compare_hu_moments(huMoments1, huMoments2):
    return cv2.matchShapes(huMoments1, huMoments2, cv2.CONTOURS_MATCH_I1, 0.0)

def main():
    image_path = input("Digite o caminho da nova imagem: ")
    image = load_image_and_remove_bg(image_path)
    if image is None:
        print("Erro ao processar a imagem ou imagem não suportada.")
        return
    mask = create_mask(image)
    contours = extract_contours(mask)

    hu_moments_file = input("Digite o caminho para o arquivo hu_moments1.pkl: ")
    saved_huMoments = load_hu_moments(hu_moments_file)

    best_match = float('inf')
    best_contour = None
    for contour in contours:
        huMoments = calculate_hu_moments(contour)
        similarity = compare_hu_moments(saved_huMoments, huMoments)
        if similarity < best_match:
            best_match = similarity
            best_contour = contour

    if best_contour is not None:
       
        cv2.drawContours(image, [best_contour], -1, (0, 255, 0), 3)
        cv2.putText(image, f"Similarity: {best_match:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Best Match", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No significant contour found.")

if __name__ == "__main__":
    main()
