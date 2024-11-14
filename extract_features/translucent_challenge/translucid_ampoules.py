import cv2
import numpy as np

# Carregar a imagem
image = cv2.imread('.\\frames\\canaleta_azul\\img_10_009.jpg')

# Converter para espaço de cor HSV para melhor segmentação de cor
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definir faixas de cores para a ampola (ajustar conforme necessário)
lower_bound = np.array([113, 4, 168])  # Exemplo: valor mais claro
upper_bound = np.array([145, 24, 202])

# Criar máscara
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Aplicar a máscara à imagem original
segmented_image = cv2.bitwise_and(image, image, mask=mask)

# Converter a imagem segmentada para escala de cinza
gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

# # Detecção de bordas usando Canny na imagem segmentada
# edges = cv2.Canny(gray_segmented, 50, 150, apertureSize=3)

# # Aplicar um threshold para detectar áreas brilhantes
# circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for i in circles[0, :]:
#         # desenhar o círculo externo
#         cv2.circle(segmented_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
#         # desenhar o centro do círculo
#         cv2.circle(segmented_image, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('Detected Circles', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Aplicar filtro de Sobel para encontrar gradientes
sobelx = cv2.Sobel(gray_segmented, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray_segmented, cv2.CV_64F, 0, 1, ksize=5)

# Calcular a magnitude dos gradientes
magnitude = np.sqrt(sobelx**2 + sobely**2)

cv2.imshow('Edge Magnitude', magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()

