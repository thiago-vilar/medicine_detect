import pickle
import matplotlib.pyplot as plt
import numpy as np

def main():
    filename = input("Digite o caminho para o arquivo de assinatura do contorno (.pkl): ")
    try:
        with open(filename, 'rb') as f:
            assinatura = pickle.load(f)

        # Converter assinatura em um array numpy, caso ainda não seja
        assinatura = np.array(assinatura)

        # Remover dimensões desnecessárias
        assinatura = np.squeeze(assinatura)

        # Verificar o novo shape
        print(f"Shape da assinatura após squeeze: {assinatura.shape}")

        # Verificar se o shape é (N, 2)
        if assinatura.shape[1] != 2:
            print("Dados inesperados. Esperado um array de coordenadas com shape (N, 2).")
            return

        # Separar as coordenadas x e y
        x = assinatura[:, 0]
        y = assinatura[:, 1]

        # Plotar o contorno
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, '-o', markersize=2)
        plt.title('Assinatura do Contorno')
        plt.xlabel('Eixo X')
        plt.ylabel('Eixo Y')

        # Inverter o eixo Y
        plt.gca().invert_yaxis()

        plt.axis('equal')  # Para manter a proporção dos eixos
        plt.show()

    except Exception as e:
        print(f"Ocorreu um erro ao carregar ou exibir a assinatura: {e}")

if __name__ == "__main__":
    main()
