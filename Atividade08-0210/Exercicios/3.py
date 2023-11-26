import cv2
import numpy as np
import matplotlib.pyplot as plt

#Colocando as imagens 
path_f = './imagem/f.jpeg   '
image_f = cv2.imread(path_f, cv2.IMREAD_GRAYSCALE)

# Definindo o elemento 
se_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

# Erosão seguida de dilatação
img_eroded = cv2.erode(image_f, se_rect)
img_dilated = cv2.dilate(img_eroded, se_rect)

# Exibição das imagens
plt.figure(figsize=(15, 5))

# Subplot 1: Imagem original
plt.subplot(1, 3, 1)
plt.imshow(image_f, cmap='gray')
plt.title('f) Original')
plt.axis('off')

# Subplot 2: Imagem após erosão
plt.subplot(1, 3, 2)
plt.imshow(img_eroded, cmap='gray')
plt.title('Após Erosão')
plt.axis('off')

# Subplot 3: Imagem após dilatação
plt.subplot(1, 3, 3)
plt.imshow(img_dilated, cmap='gray')
plt.title('Após Dilatação')
plt.axis('off')

plt.tight_layout()
plt.show()