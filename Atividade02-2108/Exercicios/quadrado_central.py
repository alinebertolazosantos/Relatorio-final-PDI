import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage


#Função de exibição de comparação de imagens
def print_image(original, rect):
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Retângulo no centro")
    plt.imshow(rect, cmap="gray")
    plt.show()

#Função que centraliza o quadrado na figura
def rect_on_center(array):
    shape = array.shape
    shape = (shape[0] // 2, shape[1] // 2)
    array[shape[0] - 15 : shape[0] + 15, shape[1] - 15 : shape[1] + 15] = 0
    return array

#Função principal 
def main():
    path = "C:/Users/Berto/Documents/Facul/Processamento_Digital_Imagem/image"

    #variaveis recebem o endereço das imagens
    lena = np.array(Image.open('../image/lena_gray_512.tif'))
    cameraman = np.array(Image.open('../image/cameraman.tif'))
    house = np.array(Image.open('../image/casa.tif'))

    # Conversão para array
    lena_array = np.array(lena)
    house_array = np.array(house)
    cameraman_array = np.array(cameraman)

    # retângulo branco no centro
    lena_intensity = rect_on_center(lena_array.copy())
    donout_intensity = rect_on_center(house_array.copy())
    cameraman_intensity = rect_on_center(cameraman_array.copy())

    # Exibição com matplotlib - original e com retângulo preto no centro
    # Lena
    plt.figure("Lena")
    print_image(lena_array, lena_intensity)

    # house
    plt.figure("house")
    print_image(house_array, donout_intensity)

    # Cameraman
    plt.figure("Cameraman")
    print_image(cameraman_array, cameraman_intensity)

if __name__ == "__main__":
    main()