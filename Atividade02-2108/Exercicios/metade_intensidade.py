import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

def main():
    #variaveis recebem o endereço das imagens
    lena = np.array(Image.open('../image/lena_gray_512.tif'))
    cam = np.array(Image.open('../image/cameraman.tif'))
    house = np.array(Image.open('../image/casa.tif'))

    #Intensidade da idade cortada na metade
    npLenaPixelImage =  (lena / 2).astype(int)
    npCamPixelImage =  (cam / 2).astype(int)
    npHousePixelImage =  (house / 2).astype(int)

    #Exibir as figuras
    fig = plt.figure(figsize=(10, 5))
    plt1 = plt.subplot(1, 3, 1)
    plt2 = plt.subplot(1, 3, 2)
    plt3 = plt.subplot(1, 3, 3)
    plt1.title.set_text("Lena Com Intensidade Reduzida")
    plt2.title.set_text("Cameraman Com Intensidade Reduzida")
    plt3.title.set_text("House Com Intensidade Reduzida")
    plt.subplots_adjust(wspace=0.5)

    plt1.imshow(npLenaPixelImage, cmap="gray")
    plt2.imshow(npCamPixelImage, cmap="gray")
    plt3.imshow(npHousePixelImage, cmap="gray")
    plt.show()

if __name__ == "__main__":
    main()