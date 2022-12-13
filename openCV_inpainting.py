import numpy as np
import cv2 as cv
import math

img1 = cv.imread('C:/Users/jonas/Desktop/mcgill/Fourth year/Comp 558/Final project/FMM_image_inpainting/Results/Result_5_e3_US.png',0)
img2 = cv.imread('C:/Users/jonas/Desktop/mcgill/Fourth year/Comp 558/Final project/FMM_image_inpainting/Results/Result_5_e3_Telea.png',0)


#height, width , w = img.shape
# img = cv.resize(img, (math.floor(width/5), math.floor(height/5)))
# mask = cv.resize(mask, (math.floor(width/5), math.floor(height/5)))


dst = cv.subtract(img1,img2)
# dst = cv.inpaint(img,mask,3,cv.INPAINT_NS)

cv.imwrite('C:/Users/jonas/Desktop/mcgill/Fourth year/Comp 558/Final project/FMM_image_inpainting/Results/Result_5_UT.png',dst)

cv.imshow('dst',dst)
cv.waitKey(0)
cv.destroyAllWindows()
