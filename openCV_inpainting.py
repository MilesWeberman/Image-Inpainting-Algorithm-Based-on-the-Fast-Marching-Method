import numpy as np
import cv2 as cv

img = cv.imread('/Users/milesweberman/Desktop/inpainting/input_img1.png')
mask = cv.imread('/Users/milesweberman/Desktop/inpainting/mask1.png',0)

print(img.shape)

dst = cv.inpaint(img,mask,3,cv.INPAINT_TELEA)

cv.imshow('dst',dst)
cv.waitKey(0)
cv.destroyAllWindows()
