import cv2
import numpy as np
import pickle

def load_contour_signature(filename):
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"Erro: O arquivo {filename} não foi encontrado.")
        return None

def preprocess_image(gray_image):
    processed_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, thresh = cv2.threshold(processed_image, 127, 255, cv2.THRESH_BINARY)
    return thresh

def detect_contours(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_matches(color_image, contours, reference_contour):
    min_similarity = float('inf')
    best_match = None
    for contour in contours:
        contour = normalize_contour(contour)
        for angle in np.arange(0, 360, 180):
            rotated_contour = rotate_contour(contour, angle)
            for scale in [0.5, 0.75, 1, 1.25, 1.5]:
                scaled_contour = rescale_contour(rotated_contour, scale)
                similarity = cv2.matchShapes(reference_contour, scaled_contour, cv2.CONTOURS_MATCH_I1, 0.0)
                if similarity < min_similarity:
                    min_similarity = similarity
                    best_match = scaled_contour

    if best_match is not None and min_similarity < 0.05:
        cv2.drawContours(color_image, [best_match], -1, (0, 0, 255), 3)
        print(f"Contorno desenhado com similaridade: {min_similarity:.2f}")
    else:
        print("Nenhuma correspondência próxima encontrada.")

def normalize_contour(contour):
    moments = cv2.moments(contour)
    cx = int(moments['m10']/moments['m00'])
    cy = int(moments['m01']/moments['m00'])
    contour -= [cx, cy]
    return contour

def rotate_contour(contour, angle):
    angle_rad = np.radians(angle)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    return np.dot(contour, rotation_matrix)

def rescale_contour(contour, scale):
    return contour * scale

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
            find_matches(color_image, contours, reference_contour)
            cv2.imshow('Matched Contours', color_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Nenhum contorno foi detectado na imagem.")
    else:
        print("Falha ao carregar a assinatura do contorno.")

if __name__ == "__main__":
    main()
