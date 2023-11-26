import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

enhance_path = 'C:/Users/Berto/Documents/Facul/Processamento_Digital_Imagem/image/enhanceme.gif'
fractured_path = 'C:/Users/Berto/Documents/Facul/Processamento_Digital_Imagem/image/fractured.tif'

enhance_img = np.array(Image.open(enhance_path))
fractured_img = np.array(Image.open(fractured_path))

def bit_plane(img, bit):
    return (img & (1 << bit)) >> bit


# Representação de cada plano de bits
for i in range(8):
    bit_enhance = bit_plane(enhance_img, i)
    bit_Fig0308 = bit_plane(fractured_img, i)

 # Exibição
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(bit_enhance, cmap='gray')
    axs[0].set_title(f'Enhance-me (Bit Plane {i})')
    axs[1].imshow(bit_Fig0308, cmap='gray')
    axs[1].set_title(f'fractured (Bit Plane {i})')  
    plt.show()

# Equalização do histograma
equalized_enhance = cv2.equalizeHist(enhance_img)
equalized_fractured= cv2.equalizeHist(fractured_img)

# Exibição
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(equalized_enhance, cmap='gray')
axs[0].set_title('Enhance-me (Equalização)')
axs[1].imshow(equalized_fractured, cmap='gray')
axs[1].set_title('fractured (Equalização)')
plt.show()
