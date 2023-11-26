# 3  Crie um filtro passa-alta do tipo ideal, butterworth e gaussiano e aplique-o às imagens disponibilizadas.

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

def fourier_spectrum(image):
 # Computa a transformada de Fourier 2D
 f = np.fft.fft2(image)
 # Centraliza as frequências baixas
 fshift = np.fft.fftshift(f)
 # Calcula a magnitude e aplica o logaritmo para melhor visualização
 magnitude_spectrum = np.log(np.abs(fshift) + 1)
 return magnitude_spectrum


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

def ideal_highpass_filter(image, cutoff):
    return 1 - ideal_lowpass_filter(image, cutoff)

def butterworth_highpass_filter(image, cutoff, order=2):
    return 1 - butterworth_lowpass_filter(image, cutoff, order)

def gaussian_highpass_filter(image, cutoff):
    return 1 - gaussian_lowpass_filter(image, cutoff)

# Função para aplicar o filtro usando transformada de Fourier
def apply_filter(image, filter):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fshift = fshift * filter
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

images = [sinc_original, sinc_rot, sinc_rot2, sinc_trans]
titles = ['Original', 'Rotacionada (40º)', 'Rotacionada (20º)', 'Transladada']

for idx, image in enumerate(images):
    # Fourier
    spectrum = fourier_spectrum(image)
    
    # Criação dos filtros
    ideal_hp = ideal_highpass_filter(image, cutoff)
    butter_hp = butterworth_highpass_filter(image, cutoff)
    gaussian_hp = gaussian_highpass_filter(image, cutoff)

    # Aplicação dos filtros
    result_ideal = apply_filter(image, ideal_hp)
    result_butter = apply_filter(image, butter_hp)
    result_gaussian = apply_filter(image, gaussian_hp)
    
    # Exibição
    plt.figure(figsize=(20, 10))

    # Imagem original
    plt.subplot(4, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'a) Imagem {titles[idx]}')
    plt.axis('off')

    # Espectro de Fourier
    plt.subplot(4, 4, 2)
    plt.imshow(spectrum, cmap='gray')
    plt.title('b) Espectro de Fourier')
    plt.axis('off')

    # Filtro passa-alta Ideal
    plt.subplot(4, 4, 3)
    plt.imshow(ideal_hp, cmap='gray')
    plt.title('c) Filtro Ideal Passa-Alta')
    plt.axis('off')
    
    # Resultado filtro Ideal
    plt.subplot(4, 4, 4)
    plt.imshow(result_ideal, cmap='gray')
    plt.title('d) Após Filtro Ideal')
    plt.axis('off')

    # Filtro passa-alta Butterworth
    plt.subplot(4, 4, 7)
    plt.imshow(butter_hp, cmap='gray')
    plt.title('c) Filtro Butterworth Passa-Alta')
    plt.axis('off')
    
    # Resultado filtro Butterworth
    plt.subplot(4, 4, 8)
    plt.imshow(result_butter, cmap='gray')
    plt.title('d) Após Filtro Butterworth')
    plt.axis('off')

    # Filtro passa-alta Gaussiano
    plt.subplot(4, 4, 11)
    plt.imshow(gaussian_hp, cmap='gray')
    plt.title('c) Filtro Gaussiano Passa-Alta')
    plt.axis('off')
    
    # Resultado filtro Gaussiano
    plt.subplot(4, 4, 12)
    plt.imshow(result_gaussian, cmap='gray')
    plt.title('d) Após Filtro Gaussiano')
    plt.axis('off')

    # Ajusta o layout e mostra a figura
    plt.tight_layout()
    plt.show()