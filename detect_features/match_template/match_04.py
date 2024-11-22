import numpy as np
import cv2

# Ler a imagem original e o template
img_original = cv2.imread('.\\frames\\img_0_010.jpg')
template_original = cv2.imread('.\\features\\medicine_png\\medicine_2.png', cv2.IMREAD_UNCHANGED)  # Carrega com canal alfa se disponível

# Redimensionar a imagem original e o template para corresponder ao fator de escala da imagem
img = cv2.resize(img_original, (0, 0), fx=0.8, fy=0.8)  # Adjust the scaling factor as needed
template = cv2.resize(template_original, (0, 0), fx=0.8, fy=0.8)  # Adjust the scaling factor as needed
h, w = template.shape[:2]

# Métodos de detecção
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
           cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

for method in methods:
    result = cv2.matchTemplate(img, template[:, :, :3], method)  # Usar apenas os canais BGR do template
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Definir um limite de confiança para aceitar matches (ajuste este threshold conforme necessário)
    threshold = 0.8  # Example threshold
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        if min_val > (1 - threshold):
            continue
        top_left = min_loc
    else:
        if max_val < threshold:
            continue
        top_left = max_loc
    
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    # Preparar a imagem de resultado usando a imagem original em cores
    img_display = img.copy()

    # Sobrepor o template colorido na imagem na localidade exata do match
    if template_original.shape[2] == 4:  # Se houver canal alfa
        mask = template[:, :, 3] / 255.0
        overlay = template[:, :, :3]
        for c in range(3):  # Sobrepor cada canal, respeitando o alfa
            img_display[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], c] = (
                overlay[:, :, c] * mask + img_display[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], c] * (1 - mask)
            ).astype(np.uint8)

    # Mostrar o resultado
    cv2.imshow(f'Result using {method}', img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
