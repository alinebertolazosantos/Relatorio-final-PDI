import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#enhance_path = 'C:/Users/Berto/Documents/Facul/Processamento_Digital_Imagem/image/enhanceme.gif'
enhance_path = 'C:/Users/Berto/Documents/Facul/Processamento_Digital_Imagem/image/fractured.tif'
img = cv2.imread(enhance_path, cv2.IMREAD_GRAYSCALE)
# Usando PIL para abrir a imagem
image_pil = Image.open(enhance_path)
# Convertendo a imagem PIL para um array numpy
img = np.array(image_pil)

def power_transform(c, gamma, img):
 return c * np.power(img, gamma)
gamma_values = [0.1, 0.5, 1, 2, 5]
for gamma in gamma_values:
 
 transformed_img = power_transform(1, gamma, img)
 plt.imshow(transformed_img, cmap='gray')
 plt.title(f' Y = {gamma}')
 plt.show()