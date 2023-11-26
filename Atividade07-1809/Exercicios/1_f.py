#1 - f) Aplique uma translação nos eixos x e y no quadrado e repita os passo b-d

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Imagem 512x512
imagem = np.zeros((512, 512), dtype=np.uint8)

# quadrado branco
cv2.rectangle(imagem, (204, 204), (308, 308), 255, -1)

# Aplicando translação de 40 pixels nos eixos x e y na imagem original
translacao = np.float32([[1, 0, 40], [0, 1, 40]])

imagem_transladada = cv2.warpAffine(imagem, translacao, (512, 512))

#  Calculando o espectro de Fourier (amplitudes) da imagem transladada
f_trans = np.fft.fft2(imagem_transladada)
fshift_trans = np.fft.fftshift(f_trans)
magnitude_spectrum_trans = 20*np.log(np.abs(fshift_trans) + 1) 

# Calculando o espectro de Fourier (fases) da imagem transladada
fase_spectrum_trans = np.angle(fshift_trans)

# Usando subplots para exibir as imagens lado a lado
fig, axs = plt.subplots(1, 3, figsize=(15,5))

# Imagem transladada
axs[0].imshow(imagem_transladada, cmap='gray')
axs[0].set_title('Imagem Transladada')
axs[0].axis('off')

# Espectro de Amplitude
axs[1].imshow(magnitude_spectrum_trans, cmap='gray')
axs[1].set_title('Espectro de Fourier - Amplitude')
axs[1].axis('off')

# Espectro de Fase
axs[2].imshow(fase_spectrum_trans, cmap='gray')
axs[2].set_title('Espectro de Fourier - Fase')
axs[2].axis('off')
plt.tight_layout()
plt.show()
