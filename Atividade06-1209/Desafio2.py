import cv2
import numpy as np


# Read video
vrObj = cv2.VideoCapture('./imagem/surveillance.mpg')
frame_width = int(vrObj.get(3))
frame_height = int(vrObj.get(4))
vwObj =cv2.VideoWriter('Background_Subtraction.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (frame_width, frame_height))
nFrames = int(vrObj.get(cv2.CAP_PROP_FRAME_COUNT))

# Perform background accumulation and subtraction
alpha = 0.95
theta = 0.1
_, background = vrObj.read()
background = cv2.cvtColor(background,
cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
while True:
    ret, frame = vrObj.read()
    if not ret:
        break
    currImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    
    background = alpha * background + (1 - alpha) * currImg

    diffImg = np.abs(currImg - background)
    threshImg = (diffImg > theta).astype(np.uint8) * 255
    cv2.imshow('New frame', currImg)
    cv2.imshow('Background frame', background)
    cv2.imshow('Difference image', diffImg)
    cv2.imshow('Thresholded difference image', threshImg)

    vwObj.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vrObj.release()
vwObj.release()
cv2.destroyAllWindows()
# Save images
cv2.imwrite('Background_Subtraction_curr.png',
currImg * 255.0)
cv2.imwrite('Background_Subtraction_background.png', background * 255.0) 
cv2.imwrite('Background_Subtraction_thresh.png', threshImg)