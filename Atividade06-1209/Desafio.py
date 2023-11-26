import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load test images
origImg = cv2.imread('./Imagem/pcbCropped.png', cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
defectImg = cv2.imread('./Imagem/pcbCroppedTranslatedDefected.png', cv2.IMREAD_GRAYSCALE).astype(np.float64) /255.0

# Perform shift
xShift = 10
yShift = 10
row, col = origImg.shape
registImg = np.zeros_like(defectImg)
registImg[yShift:row, xShift:col] = defectImg[0:row - yShift, 0:col - xShift]

# Show difference images
diffImg1 = np.abs(origImg - defectImg)
plt.subplot(1, 3, 1), plt.imshow(diffImg1,
cmap='gray'), plt.title('Unaligned DifferenceImage')
diffImg2 = np.abs(origImg - registImg)
plt.subplot(1, 3, 2), plt.imshow(diffImg2,
cmap='gray'), plt.title('Aligned Difference Image')
bwImg = diffImg2 > 0.15
height, width = bwImg.shape
border = round(0.05 * width)
borderMask = np.zeros_like(bwImg)
borderMask[border:height-border, border:width - border] = 1 
bwImg = bwImg * borderMask
plt.subplot(1, 3, 3), plt.imshow(bwImg,
cmap='gray'), plt.title('Thresholded + AlignedDifference Image')

# Save images
cv2.imwrite('Defect_Detection_diff.png', diffImg1 * 255.0)
cv2.imwrite('Defect_Detection_diffRegisted.png', diffImg2 * 255.0)
cv2.imwrite('Defect_Detection_bw.png', bwImg * 255.0)
plt.show()