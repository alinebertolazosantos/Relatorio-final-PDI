import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

def main():
    #variaveis recebem o endereço das imagens
    lena = np.array(Image.open('../image/lena_gray_512.tif'))
    cam = np.array(Image.open('../image/cameraman.tif'))
    house = np.array(Image.open('../image/casa.tif'))

    #Transforma a foto da lena em negativa
    npLenaNegative = lena
    npLenaNegative = 255 - npLenaNegative

    #Transforma a foto do cameraman em negativa
    npCamNegative = cam
    npCamNegative = 255 - npCamNegative

    #Transforma a foto da house em negativa
    npHouseNegative = house
    npHouseNegative = 255 - npHouseNegative

    #Preparando para exibir as imagens pelo matplotlib
    fig = plt.figure(figsize=(10, 5))
    plt1 = plt.subplot(1, 3, 1)
    plt1.title.set_text("Lena Negativa")
    plt2 = plt.subplot(1, 3, 2)
    plt2.title.set_text("Cameraman Negativa")
    plt3 = plt.subplot(1, 3, 3)
    plt3.title.set_text("House Negativa")

    plt.subplots_adjust(wspace=0.5)
    plt1.imshow(npLenaNegative, cmap="gray")
    plt2.imshow(npCamNegative, cmap="gray")
    plt3.imshow(npHouseNegative, cmap="gray")
    plt.show()

if __name__ == "__main__":
    main()