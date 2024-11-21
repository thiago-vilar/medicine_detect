import cv2
import numpy as np

# Ler a imagem original e o template

img_original = cv2.imread('.\\frames\\img_0_010.jpg')
template_original = cv2.imread('.\\features\\medicine_png\\medicine_1.png', cv2.IMREAD_UNCHANGED)  

img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
img_gray = cv2.resize(img_gray, (0, 0), fx=0.8, fy=0.8)
img_gray_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)


template = cv2.resize(template_original, (0, 0), fx=0.8, fy=0.8)
h, w = template.shape[:2]

# Checar se o template possui canal alfa
if template.shape[2] == 4:
    alpha_channel = template[:, :, 3]
    template = template[:, :, :3]  
else:
    alpha_channel = np.ones((h, w), dtype=template.dtype) * 255 

alpha_channel = alpha_channel / 255.0 

# Métodos de detecção
methods = [cv2.TM_CCOEFF_NORMED]

for method in methods:
    result = cv2.matchTemplate(img_gray, cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), method)
    _, _, min_loc, max_loc = cv2.minMaxLoc(result)

    # Usar max_loc ou min_loc dependendo do método
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Mesclar o template colorido com a imagem em tons de cinza usando o canal alfa
    for c in range(3):
        img_gray_color[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], c] = \
            alpha_channel * template[:, :, c] + \
            (1 - alpha_channel) * img_gray_color[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], c]

    # Mostrar o resultado
    cv2.imshow('Result', img_gray_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
               