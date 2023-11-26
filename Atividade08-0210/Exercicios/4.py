import cv2
import numpy as np
import matplotlib.pyplot as plt

# Imagem G
img_g = cv2.imread('./imagem/g.jpeg', cv2.IMREAD_GRAYSCALE)

# Elemento estruturante (se_disk)
se_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Dilatação para melhorar o texto
img_processed = cv2.dilate(img_g, se_disk)

# Exibindo a imagem original e a imagem processada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_g, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_processed, cmap='gray')
plt.title('Imagem Processada')
plt.axis('off')
plt.tight_layout()
plt.show()
