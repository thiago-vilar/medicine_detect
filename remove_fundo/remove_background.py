# Importar os módulos necessários
from rembg import remove
from PIL import Image
import os
from datetime import datetime

# Função para remover o fundo de uma imagem
def remove_background(input_path):
    """Remove o fundo da imagem usando a biblioteca rembg e salva a saída na pasta 'frame'."""
    
    # Verifica se a imagem de entrada existe
    if not os.path.exists(input_path):
        print(f"❌ Erro: A imagem {input_path} não foi encontrada.")
        return

    # Abrir a imagem de entrada
    input_image = Image.open(input_path)

    # Remover o fundo da imagem
    output_image = remove(input_image)

    # Criar a pasta de saída 'frame' se ela não existir
    output_dir = './frame'
    os.makedirs(output_dir, exist_ok=True)

    # Gerar o nome do arquivo de saída com base na data e hora atual
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'img_{timestamp}.png')

    # Salvar a imagem processada no caminho de saída
    output_image.save(output_path)

    print(f"✔️ Imagem sem fundo salva em: {output_path}")

# Programa principal
if __name__ == "__main__":
    # Solicitar o caminho da imagem de entrada ao usuário
    input_path = input("Digite o caminho da imagem de entrada: ")

    # Remover o fundo da imagem
    remove_background(input_path)
