import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Modificado de https://www.cin.ufpe.br/~cabm/visao/Aula08_PercepcaoCores.pdf
def normalize(channel, max_, min_):
    output = (channel - min_) / (max_ - min_)
    return output

def white_patch(image, white_threshold):
    out = image / 255.0
    for channel in range(image.shape[2]):
        flat = out[:, :, channel].ravel()
        flat.sort()
        min_ = flat[int(flat.shape[0] * (1 - white_threshold))]
        max_ = flat[int(flat.shape[0] * white_threshold)]
        out[:, :, channel] = (out[:, :, channel] - min_) / (max_ - min_ + 1e-5)
    out = np.clip(out, 0, 1)
    return out

def gray_world(image, white_threshold):
    out = image / 255.0
    for channel in range(image.shape[2]):
        mean_ = out[:, :, channel].mean()
        out[:, :, channel] = out[:, :, channel] / mean_
        flat = out[:, :, channel].ravel()
        flat.sort()
        max_ = flat[int(flat.shape[0] * white_threshold)]
        out[:, :, channel] = out[:, :, channel] / max_
    out = np.clip(out, 0, 1)
    return out

def angular_error(image_A, image_B):
    XYZ_A = cv2.cvtColor(image_A, cv2.COLOR_BGR2XYZ) / 255.0
    XYZ_B = cv2.cvtColor(image_B, cv2.COLOR_BGR2XYZ) / 255.0
    norm_A = np.linalg.norm(XYZ_A, axis=-1)
    norm_B = np.linalg.norm(XYZ_B, axis=-1)
    zero_norms_mask = (norm_A == 0) | (norm_B == 0)
    dot_product = np.sum(XYZ_A * XYZ_B, axis=-1)
    cos_theta = np.clip(dot_product / (norm_A * norm_B + 1e-5), -1, 1)
    error = np.arccos(cos_theta)
    error[zero_norms_mask] = 0.0
    angular_errors = np.degrees(error)
    mean_angular_error = np.mean(angular_errors)
    return mean_angular_error

def save_image(output_image, base_path, index):
    folder_path = os.path.dirname(base_path)
    save_path = os.path.join(folder_path, f"color_c{index}.jpg")
    cv2.imwrite(save_path, cv2.cvtColor((output_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"Imagem salva em: {save_path}")

if __name__ == "__main__":
    image_paths = ["./frames/new_test_light/thiago_fotos_10_down_lighton_ampoules/img_1_009.jpg"]
    n_images = len(image_paths)

    fig, ax = plt.subplots(1, 3, figsize=(20, 14), subplot_kw=dict(xticks=[], yticks=[]))
    folder_name = os.path.basename(os.path.dirname(image_paths[0]))
    print(folder_name)
    
    image = cv2.cvtColor(cv2.imread(image_paths[0]), cv2.COLOR_BGR2RGB)
    ax[0].imshow(image)
    ax[0].set_title('Input Image')

    wp = white_patch(image, 0.96)
    ax[1].imshow(wp)
    mae = angular_error(image, (255 * wp).astype(np.uint8))
    ax[1].set_title(f"White Patch applied\nMean angular error: {mae:.2f} degrees")
    save_image(wp, image_paths[0], 1)  

    gw = gray_world(image, 0.95)
    ax[2].imshow(gw)
    mae = angular_error(image, (255 * gw).astype(np.uint8))
    ax[2].set_title(f"Gray World applied\nMean angular error: {mae:.2f} degrees")
    save_image(gw, image_paths[0], 2)  

    plt.show()
