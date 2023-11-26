import cv2
import numpy as np
import matplotlib.pyplot as plt

#imagem h)
img_h = cv2.imread('./imagem/h.jpeg', cv2.IMREAD_GRAYSCALE)

# Elemento estruturante (se_disk)
se_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Erosão para extrair a borda
img_eroded = cv2.erode(img_h, se_disk)

# Subtraindo a imagem erodida da imagem original para obter a borda
img_border = cv2.subtract(img_h, img_eroded)

# Exibindo a imagem original e a borda extraída
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_h, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_border, cmap='gray')
plt.title('Borda Extraída')
plt.axis('off')
plt.tight_layout()
plt.show()
