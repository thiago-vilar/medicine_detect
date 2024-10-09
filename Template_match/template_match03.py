import numpy as np
import cv2

# Ler a imagem original e o template
img_original = cv2.imread('./frames/frame_20240924_110521.jpg')
template_original = cv2.imread('./med_rm/img_20241004_104055.png')

# Converter a imagem original para tons de cinza
img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
img_gray = cv2.resize(img_gray, (0, 0), fx=0.8, fy=0.8)
img_gray_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)  # Convertendo de volta para BGR para fusão de cores

# Redimensionar o template para corresponder ao fator de escala da imagem
template = cv2.resize(template_original, (0, 0), fx=0.8, fy=0.8)
h, w, _ = template.shape

# Métodos de detecção
methods = [cv2.TM_CCOEFF_NORMED]

for method in methods:
    result = cv2.matchTemplate(img_gray, cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Usar max_loc ou min_loc dependendo do método
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Sobrepor o template colorido na imagem em tons de cinza
    img_gray_color[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = template

    # Mostrar o resultado
    cv2.imshow('Result', img_gray_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
