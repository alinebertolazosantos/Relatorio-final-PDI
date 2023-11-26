#1 - c) calcular e visualizar seu espectro de Fourier (fases)

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Imagem 512x512
imagem = np.zeros((512, 512), dtype=np.uint8)

# quadrado branco
cv2.rectangle(imagem, (204, 204), (308, 308), 255, -1)

# Transformada de Fourier
f = np.fft.fft2(imagem)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

# Calculando as fases
fase_spectrum = np.angle(fshift)

# Exibição da imagem
plt.imshow(fase_spectrum, cmap = 'gray')
plt.title('Espectro de Fourier - Fases')
plt.show()
