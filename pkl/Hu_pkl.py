import pickle
import os

def load_hu_moments(filename):
    """Carrega os Momentos de Hu de um arquivo .pkl."""
    with open(filename, 'rb') as file:
        huMoments = pickle.load(file)
    return huMoments

def display_hu_moments(huMoments):
    """Exibe os Momentos de Hu."""
    print("Momentos de Hu:")
    for index, moment in enumerate(huMoments):
        print(f"  Hu[{index + 1}]: {moment}")

def main():
    directory = input("Digite o diretório dos arquivos Momentos de Hu: ")
    filename = input("Digite o nome do arquivo .pkl para carregar (ex: hu_moment1.pkl): ")
    full_path = os.path.join(directory, filename)
    
    if not os.path.exists(full_path):
        print("O arquivo especificado não existe.")
        return

    huMoments = load_hu_moments(full_path)
    display_hu_moments(huMoments)

if __name__ == "__main__":
    main()
