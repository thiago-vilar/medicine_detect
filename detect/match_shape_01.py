import cv2
import numpy as np
import pickle
from datetime import datetime

def load_contour_signature(filename):
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"Erro: O arquivo {filename} não foi encontrado.")
        return None

def preprocess_image(gray_image):
    # Aplicar suavização para reduzir o ruído
    processed_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, thresh = cv2.threshold(processed_image, 127, 255, cv2.THRESH_BINARY)
    return thresh

def detect_contours(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def match_contour(color_image, contours, reference_contour):
    min_similarity = float('inf')
    best_match = None

    for contour in contours:
        similarity = cv2.matchShapes(reference_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0)
        if similarity < min_similarity:
            min_similarity = similarity
            best_match = contour

    if best_match is not None:
        cv2.drawContours(color_image, [best_match], -1, (0, 0, 255), 3) 
        print(f"Melhor correspondência encontrada com similaridade: {min_similarity:.2f}")
    else:
        print("Nenhuma correspondência encontrada.")

def create_zoomable_window(image, window_name="Image Viewer"):
    def update(x):
        scale = cv2.getTrackbarPos('Zoom', window_name) / 100
        scaled_img = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(window_name, scaled_img)

    cv2.namedWindow(window_name)
    cv2.createTrackbar('Zoom', window_name, 100, 400, update)
    update(0)  # Initial display
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image_path = 'frames/frame_20240924_110521.jpg'
    contour_signature_path = 'largest_contour.pkl'
    
    reference_contour = load_contour_signature(contour_signature_path)
    if reference_contour is not None:
        image = cv2.imread(image_path)
        if image is None:
            print("Erro ao carregar a imagem.")
            return
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = preprocess_image(gray_image)
        contours = detect_contours(thresh)
        color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        if contours:
            match_contour(color_image, contours, reference_contour)
            create_zoomable_window(color_image)
        else:
            print("Nenhum contorno foi detectado na imagem.")
    else:
        print("Falha ao carregar a assinatura do contorno.")

if __name__ == "__main__":
    main()
