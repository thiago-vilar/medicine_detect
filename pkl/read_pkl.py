import cv2
import pickle
import numpy as np
import os

def read_contour_signature(filename):
    if not os.path.exists(filename):
        print(f"Arquivo {filename} não encontrado.")
        return None
    with open(filename, 'rb') as file:
        return pickle.load(file)

def display_contour_signature(signature_path):
    contour = read_contour_signature(signature_path)
    if contour is None:
        print("Não foi possível carregar a assinatura do contorno.")
        return

    print(f"Formato do contorno carregado: {contour.shape}")

    # Verificar a forma do contorno e indexar adequadamente
    if len(contour.shape) == 3:
        # Forma (n_points, 1, 2)
        x = contour[:, 0, 0]
        y = contour[:, 0, 1]
    elif len(contour.shape) == 2:
        # Forma (n_points, 2)
        x = contour[:, 0]
        y = contour[:, 1]
        # Ajustar o contorno para ter a forma (n_points, 1, 2)
        contour = contour.reshape(-1, 1, 2)
    else:
        print("Formato do contorno não reconhecido.")
        return

    x_min = int(np.min(x))
    x_max = int(np.max(x))
    y_min = int(np.min(y))
    y_max = int(np.max(y))

    width = x_max - x_min + 20  # Adicionar margem
    height = y_max - y_min + 20  # Adicionar margem

    # Criar uma imagem em branco com fundo branco
    image = np.ones((height, width), dtype=np.uint8) * 255  # Fundo branco

    # Ajustar o contorno para caber na imagem
    adjusted_contour = contour.copy()
    adjusted_contour[:, 0, 0] -= x_min - 10  # Ajustar com a margem
    adjusted_contour[:, 0, 1] -= y_min - 10

    # Desenhar apenas as linhas do contorno em preto
    cv2.drawContours(image, [adjusted_contour.astype(int)], -1, 0, 2)  # Cor 0 (preto), espessura 2

    # Salvar e exibir a imagem
    cv2.imwrite('contour_signature_image.jpg', image)
    print("Imagem da assinatura do contorno salva como 'contour_signature_image.jpg'.")
    cv2.imshow('Assinatura do Contorno', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    signature_path = input("Digite o caminho para o arquivo de assinatura do contorno (.pkl): ")
    if not os.path.exists(signature_path):
        print(f"O arquivo {signature_path} não existe.")
        criar_novo = input("Deseja criar uma nova assinatura de contorno? (s/n): ").lower()
        if criar_novo == 's':
            reference_image_path = input("Digite o caminho para a imagem de referência: ")
            output_path = signature_path  # Salvar no caminho fornecido
            sucesso = create_contour_signature(reference_image_path, output_path)
            if not sucesso:
                print("Não foi possível criar a assinatura do contorno.")
                return
        else:
            print("Operação cancelada.")
            return

    display_contour_signature(signature_path)

def create_contour_signature(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Erro ao carregar a imagem de referência.")
        return False

    # Aplicar limiarização para destacar o objeto
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # Detectar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        # Escolher o maior contorno (supondo que seja o objeto de interesse)
        contour = max(contours, key=cv2.contourArea)

        # Visualizar o contorno detectado
        contour_image = np.ones_like(image) * 255  # Fundo branco
        cv2.drawContours(contour_image, [contour], -1, 0, 2)  # Desenhar em preto

        cv2.imshow('Contorno Detectado na Imagem de Referência', contour_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Salvar o contorno no arquivo .pkl
        with open(output_path, 'wb') as file:
            pickle.dump(contour, file)
        print(f"Assinatura do contorno salva em {output_path}")
        return True
    else:
        print("Nenhum contorno encontrado na imagem de referência.")
        return False

if __name__ == "__main__":
    main()
