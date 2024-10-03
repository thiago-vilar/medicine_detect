import cv2
import numpy as np

trackbars_initialized = False  # Variável global para checar se os trackbars foram inicializados

def on_trackbar_change(x):
    """Callback function that updates the image processing when trackbar values change."""
    global trackbars_initialized
    if trackbars_initialized:  # Só chama update_image quando os trackbars estiverem prontos
        update_image()

def apply_brightness_contrast(input_img, brightness=255, contrast=127):
    """Apply brightness and contrast adjustments to the input image."""
    brightness = int((brightness - 255) * (255 / 510))
    contrast = int((contrast - 127) * (127 / 254))

    if brightness != 0:
        shadow = 0 if brightness < 0 else brightness
        maximum = 255 if brightness > 0 else 255 + brightness
        alpha = (maximum - shadow) / 255
        gamma = shadow
        input_img = cv2.addWeighted(input_img, alpha, input_img, 0, gamma)

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha = f
        gamma = 127 * (1 - f)
        input_img = cv2.addWeighted(input_img, alpha, input_img, 0, gamma)

    return input_img

def update_image():
    """Update and display the image based on current trackbar positions."""
    global lh, hh, ls, hs, lv, hv, trackbar_window, original_image, max_hue, max_sat, max_val, hue_percent, sat_percent, val_percent

    # Pega as posições dos trackbars
    lh = cv2.getTrackbarPos('Low Hue', trackbar_window)
    hh = cv2.getTrackbarPos('High Hue', trackbar_window)
    ls = cv2.getTrackbarPos('Low Saturation', trackbar_window)
    hs = cv2.getTrackbarPos('High Saturation', trackbar_window)
    lv = cv2.getTrackbarPos('Low Value', trackbar_window)
    hv = cv2.getTrackbarPos('High Value', trackbar_window)
    brightness = cv2.getTrackbarPos('Brightness', trackbar_window)
    contrast = cv2.getTrackbarPos('Contrast', trackbar_window)

    # Calcula os percentuais
    hue_percent = 100 * (hh - lh) / (max_hue - lh) if max_hue - lh != 0 else 0
    sat_percent = 100 * (hs - ls) / (max_sat - ls) if max_sat - ls != 0 else 0
    val_percent = 100 * (hv - lv) / (max_val - lv) if max_val - lv != 0 else 0

    # Aplica os ajustes de brilho e contraste
    adjusted_image = apply_brightness_contrast(original_image, brightness, contrast)
    
    # Converte para HSV e aplica a máscara
    hsv_adjusted = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_adjusted, (lh, ls, lv), (hh, hs, hv))
    result = cv2.bitwise_and(adjusted_image, adjusted_image, mask=mask)

    # Exibe o percentual na imagem
    display_text = f"Hue %: {hue_percent:.2f}%, Sat %: {sat_percent:.2f}%, Val %: {val_percent:.2f}%"
    result_with_text = cv2.putText(result, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow(trackbar_window, result_with_text)

def main():
    global original_image, trackbar_window, max_hue, max_sat, max_val, trackbars_initialized
    image_path = input("Digite o caminho completo ou relativo da imagem: ")
    image = cv2.imread(image_path)
    if image is None:
        print("Erro: Não foi possível abrir a imagem. Verifique se o caminho está correto.")
        return

    image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    original_image = image.copy()
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    min_hue, min_sat, min_val = np.min(hsv_image, axis=(0, 1))
    max_hue, max_sat, max_val = np.max(hsv_image, axis=(0, 1))

    trackbar_window = 'Adjustment Trackbars'
    cv2.namedWindow(trackbar_window)

    # Criação dos trackbars
    cv2.createTrackbar('Low Hue', trackbar_window, min_hue, max_hue, on_trackbar_change)
    cv2.createTrackbar('High Hue', trackbar_window, max_hue, max_hue, on_trackbar_change)
    cv2.createTrackbar('Low Saturation', trackbar_window, min_sat, max_sat, on_trackbar_change)
    cv2.createTrackbar('High Saturation', trackbar_window, max_sat, max_sat, on_trackbar_change)
    cv2.createTrackbar('Low Value', trackbar_window, min_val, max_val, on_trackbar_change)
    cv2.createTrackbar('High Value', trackbar_window, max_val, max_val, on_trackbar_change)
    cv2.createTrackbar('Brightness', trackbar_window, 255, 510, on_trackbar_change)
    cv2.createTrackbar('Contrast', trackbar_window, 127, 254, on_trackbar_change)

    # Após os trackbars estarem criados, marcamos como inicializados
    trackbars_initialized = True
    update_image()  # Chama a função de atualização da imagem após a criação dos trackbars

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc key to exit
            break
        elif key == 32:  # Space key to print thresholds
            print(f"Thresholds HSV: Low=({lh}, {ls}, {lv}), High=({hh}, {hs}, {hv})")
            print(f"Hue %: {hue_percent:.2f}%, Sat %: {sat_percent:.2f}%, Val %: {val_percent:.2f}%")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
