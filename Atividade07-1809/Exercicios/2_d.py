#2 - d) a imagem resultante após aplicação de cada filtro
import cv2
import numpy as np
import matplotlib.pyplot as plt

sinc_original_path = './imagem/sinc_original.png'
sinc_original_menor_path = './imagem/sinc_original_menor.png'
sinc_rot_path = './imagem/sinc_rot.png'
sinc_rot2_path = './imagem/sinc_rot2.png'
sinc_trans_path = './imagem/sinc_trans.png'

# Ler as imagens
sinc_original = cv2.imread(sinc_original_path, cv2.IMREAD_GRAYSCALE)
sinc_original_menor = cv2.imread(sinc_original_menor_path, cv2.IMREAD_GRAYSCALE)
sinc_rot = cv2.imread(sinc_rot_path, cv2.IMREAD_GRAYSCALE)
sinc_rot2 = cv2.imread(sinc_rot2_path, cv2.IMREAD_GRAYSCALE)
sinc_trans = cv2.imread(sinc_trans_path, cv2.IMREAD_GRAYSCALE)

def create_filters(image, cutoff):
    # Filtro passa-baixa Ideal
    ideal_filter = ideal_lowpass_filter(image, cutoff)

    # Filtro passa-baixa Butterworth
    butter_filter = butterworth_lowpass_filter(image, cutoff)

    # Filtro passa-baixa Gaussiano
    gaussian_filter = gaussian_lowpass_filter(image, cutoff)

    return ideal_filter, butter_filter, gaussian_filter

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def ideal_lowpass_filter(image, cutoff):
    rows, cols = image.shape
    center = (rows / 2, cols / 2)
    filter = np.zeros((rows, cols))
    
    for x in range(cols):
        for y in range(rows):
            if distance((y, x), center) < cutoff:
                filter[y, x] = 1
    return filter

def butterworth_lowpass_filter(image, cutoff, order=2):
    rows, cols = image.shape
    center = (rows / 2, cols / 2)
    filter = np.zeros((rows, cols))
    
    for x in range(cols):
        for y in range(rows):
            filter[y, x] = 1 / (1 + (distance((y, x), center) / cutoff) ** (2 * order))
    return filter

def gaussian_lowpass_filter(image, cutoff):
    rows, cols = image.shape
    center = (rows / 2, cols / 2)
    filter = np.zeros((rows, cols))
    
    for x in range(cols):
        for y in range(rows):
            filter[y, x] = np.exp(-(distance((y, x), center) ** 2) / (2 * (cutoff ** 2)))
    return filter
cutoff = 30  

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)

def apply_filter(image, filter):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fshift = fshift * filter
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

# Define um cutoff para os filtros
cutoff = 50

# Cria os filtros
ideal_filter = ideal_lowpass_filter(sinc_original, cutoff)
butter_filter = butterworth_lowpass_filter(sinc_original, cutoff)
gaussian_filter = gaussian_lowpass_filter(sinc_original, cutoff)

# Aplica os filtros
img_ideal = apply_filter(sinc_original, ideal_filter)
img_butter = apply_filter(sinc_original, butter_filter)
img_gaussian = apply_filter(sinc_original, gaussian_filter)

# Visualização
fig, axs = plt.subplots(3, 3, figsize=(15,15))

# Imagens originais
axs[0, 0].imshow(sinc_original, cmap='gray')
axs[0, 0].set_title('Original')
axs[0, 1].imshow(ideal_filter, cmap='gray')
axs[0, 1].set_title('Filtro Ideal')
axs[0, 2].imshow(img_ideal, cmap='gray')
axs[0, 2].set_title('Aplicação do Filtro Ideal')

axs[1, 0].imshow(sinc_original, cmap='gray')
axs[1, 0].set_title('Original')
axs[1, 1].imshow(butter_filter, cmap='gray')
axs[1, 1].set_title('Filtro Butterworth')
axs[1, 2].imshow(img_butter, cmap='gray')
axs[1, 2].set_title('Aplicação do Filtro Butterworth')

axs[2, 0].imshow(sinc_original, cmap='gray')
axs[2, 0].set_title('Original')
axs[2, 1].imshow(gaussian_filter, cmap='gray')
axs[2, 1].set_title('Filtro Gaussiano')
axs[2, 2].imshow(img_gaussian, cmap='gray')
axs[2, 2].set_title('Aplicação do Filtro Gaussiano')

for ax in axs.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()