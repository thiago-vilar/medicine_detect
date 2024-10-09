import cv2
import numpy as np

def empty(a):
    pass

# Configuração de janelas e trackbars
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 23, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 20, 255, empty)
cv2.createTrackbar("Area", "Parameters", 5000, 30000, empty)

# Função para empilhar imagens para visualização
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        ver = np.vstack([np.hstack(imgArray[x]) for x in range(rows)])
    else:
        ver = np.hstack(imgArray)
    return ver

# Função para detectar e desenhar contornos
def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            cv2.drawContours(imgContour, [cnt], -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

# Escolha de modo de entrada: Webcam ou imagem estática
use_webcam = input("Usar webcam? (s/n): ").lower() == 's'
if use_webcam:
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
else:
    image_path = input("Digite o caminho da imagem: ")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Não foi possível carregar a imagem.")

while True:
    if use_webcam:
        success, img = cap.read()
        if not success:
            break
    imgContour = img.copy()
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    getContours(imgDil, imgContour)
    imgStack = stackImages(0.8, ([img, imgCanny], [imgDil, imgContour]))
    cv2.imshow("Result", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if use_webcam:
    cap.release()
cv2.destroyAllWindows()
