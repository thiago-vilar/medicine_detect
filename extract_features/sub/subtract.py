import cv2
import numpy as np


image_with_object = cv2.imread('.\\frames\\cabm\\thiago_fotos_cabm_5\\img_0_007.jpg')
image_without_object = cv2.imread('.\\frames\\cabm\\thiago_fotos_cabm_6\\img_0_006.jpg')


if image_with_object.shape != image_without_object.shape:
    print("As imagens devem ter o mesmo tamanho e a mesma perspectiva para a subtração.")
    exit()


image_with_object_gray = cv2.cvtColor(image_with_object, cv2.COLOR_BGR2GRAY)
image_without_object_gray = cv2.cvtColor(image_without_object, cv2.COLOR_BGR2GRAY)


difference = cv2.subtract(image_without_object_gray, image_with_object_gray)


_, difference = cv2.threshold(difference, 15, 255, cv2.THRESH_BINARY)



cv2.imshow('Objeto isolado', difference)


cv2.waitKey(0)
cv2.destroyAllWindows()
