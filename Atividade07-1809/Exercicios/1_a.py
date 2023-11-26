# 1 - a) crie e visualize uma imagem simples – quadrado branco sobre fundo preto;

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Imagem 512x512
imagem = np.zeros((512, 512), dtype=np.uint8)

# quadrado branco
cv2.rectangle(imagem, (204, 204), (308, 308), 255, -1)

# exibindo imagem 
plt.imshow(imagem, cmap='gray')
plt.title("Imagem Original")
plt.show()