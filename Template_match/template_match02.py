import numpy as np
import cv2

# Ler imagens e converter para escala de cinza
img_original = cv2.imread('./frames/frame_20240924_110521.jpg')
template_original = cv2.imread('./med_rm/img_20241004_104055.png')

# Pré-processamento com equalização de histograma
img = cv2.equalizeHist(cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY))
template = cv2.equalizeHist(cv2.cvtColor(template_original, cv2.COLOR_BGR2GRAY))

# Redimensionamento
img = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
template = cv2.resize(template, (0, 0), fx=0.8, fy=0.8)
h, w = template.shape

methods = [cv2.TM_CCOEFF_NORMED]

for method in methods:
    img2 = img.copy()
    result = cv2.matchTemplate(img2, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if method == cv2.TM_SQDIFF_NORMED:
        location = min_loc
    else:
        location = max_loc

    bottom_right = (location[0] + w, location[1] + h)
    cv2.rectangle(img2, location, bottom_right, 255, 2)
    cv2.imshow('Match', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
