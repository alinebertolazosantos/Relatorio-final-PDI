#2 - a) a imagem inicial;

import cv2
import numpy as np
import matplotlib.pyplot as plt

sinc_original_path = './imagem/sinc_original.png'
sinc_original_menor_path = './imagem/sinc_menor.png'
sinc_rot_path = './imagem/sinc_rot.png'
sinc_rot2_path = './imagem/sinc_rot2.png'
sinc_trans_path = './imagem/sinc_trans.png'

# Ler as imagens
sinc_original = cv2.imread(sinc_original_path, cv2.IMREAD_GRAYSCALE)
sinc_original_menor = cv2.imread(sinc_original_menor_path, cv2.IMREAD_GRAYSCALE)
sinc_rot = cv2.imread(sinc_rot_path, cv2.IMREAD_GRAYSCALE)
sinc_rot2 = cv2.imread(sinc_rot2_path, cv2.IMREAD_GRAYSCALE)
sinc_trans = cv2.imread(sinc_trans_path, cv2.IMREAD_GRAYSCALE)

# Crie uma figura para organizar as imagens e legendas
plt.figure(figsize=(15, 5))

# Imagem Original
plt.subplot(151)
plt.imshow(sinc_original, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

# Rotacionada (40º)
plt.subplot(152)
plt.imshow(sinc_rot, cmap='gray')
plt.title('Imagem Rotacionada (40º)')
plt.axis('off')

# Rotacionada (20º)
plt.subplot(153)
plt.imshow(sinc_rot2, cmap='gray')
plt.title('Imagem Rotacionada (20º)')
plt.axis('off')

# Transladada
plt.subplot(154)
plt.imshow(sinc_trans, cmap='gray')
plt.title('Imagem Transladada')
plt.axis('off')
plt.tight_layout()
plt.show()
