import cv2 as cv
import numpy as np

inital_img = cv.imread('flapbird_initial_img.PNG', cv.IMREAD_UNCHANGED)
bird = cv.imread('flapbird_bird.PNG', cv.IMREAD_UNCHANGED)


result = cv.matchTemplate(inital_img, bird, cv.TM_CCOEFF_NORMED)

#get the best match position
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)



cv.imshow("Result", result)
cv.waitKey(0)