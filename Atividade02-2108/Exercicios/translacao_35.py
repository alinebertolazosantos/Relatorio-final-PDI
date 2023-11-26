import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

def main():
    #variaveis recebem o endereço das imagens
    lena = np.array(Image.open('../image/lena_gray_512.tif'))
    cam = np.array(Image.open('../image/cameraman.tif'))
    house = np.array(Image.open('../image/casa.tif'))

    translacao_lena = lena.copy()
    translacao_cameraman = cam.copy()
    translacao_house = house.copy()
    shift_x = 35 
    shift_y = 45
    shift_vector = [shift_y, shift_x]

    translated_image_lena = ndimage.shift(translacao_lena, shift_vector, mode='constant', cval=0)
    translated_image_cam = ndimage.shift(translacao_cameraman , shift_vector, mode='constant', cval=0)
    translated_image_house = ndimage.shift(translacao_house, shift_vector, mode='constant', cval=0)

    fig = plt.figure(figsize=(10, 5))
    plt1 = plt.subplot(1, 3, 1)
    plt2 = plt.subplot(1, 3, 2)
    plt3 = plt.subplot(1, 3, 3)
    plt1.title.set_text("Lena Translação x=35 y=45")
    plt2.title.set_text("Cameraman  Translação x=35 y=45")
    plt3.title.set_text("House Translação x=35 y=45")
    plt.subplots_adjust(wspace=0.5)

    plt1.imshow(translated_image_lena, cmap="gray")
    plt2.imshow(translated_image_cam, cmap="gray")
    plt3.imshow(translated_image_house, cmap="gray")

    plt.show()

if __name__ == "__main__":
    main()