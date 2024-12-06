import cv2

def subtract_images(image_path1, image_path2):
    # Carrega as duas imagens
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    if image1 is None or image2 is None:
        print("Uma ou ambas as imagens não puderam ser carregadas.")
        return

    # Garante que ambas as imagens são do mesmo tamanho
    if image1.shape != image2.shape:
        print("As imagens têm dimensões diferentes e não podem ser subtraídas diretamente.")
        return

    # Realiza a subtração das imagens
    subtracted_image = cv2.subtract(image1, image2)

    # Salva ou mostra a imagem resultante
    result_path = 'resultado_subtracao.png'
    cv2.imwrite(result_path, subtracted_image)
    print(f"A imagem resultante foi salva como {result_path}")

    # Mostra a imagem resultante
    cv2.imshow('Imagem Resultante', subtracted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Define os caminhos das imagens
    path1 = '.\\frames\\cabm\\thiago_fotos_cabm_5\\img_0_007.jpg'
    path2 = '.\\frames\\cabm\\thiago_fotos_cabm_6\\img_0_005.jpg'

    # Chama a função para subtrair as imagens
    subtract_images(path1, path2)
