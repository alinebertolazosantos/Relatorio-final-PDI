#1 - d)   obter e visualizar seu espectro de Fourier centralizado

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Imagem 512x512
imagem = np.zeros((512, 512), dtype=np.uint8)

# quadrado branco
cv2.rectangle(imagem, (204, 204), (308, 308), 255, -1)

# Transformada de Fourier
f = np.fft.fft2(imagem)
fshift = np.fft.fftshift(f)  # Centralizando o espectro
magnitude_spectrum_centered = 20 * np.log(np.abs(fshift) + 1)  # Adicionamos 1 para evitar log(0)

# Exibição da imagem 
plt.imshow(magnitude_spectrum_centered, cmap='gray')
plt.title('Espectro de Fourier Centralizado')
plt.show()
