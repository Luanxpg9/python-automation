import cv2 as cv
import numpy as np
from os.path import abspath


#print_path = abspath('.\data\\flapbird_initial_img.PNG')
print_path = abspath('.\data\\print.png')
bird_path = abspath('.\data\\flapbird_bird.PNG')

print(bird_path)

inital_img = cv.imread(print_path, cv.IMREAD_REDUCED_COLOR_2)
bird = cv.imread(bird_path, cv.IMREAD_REDUCED_COLOR_2)


result = cv.matchTemplate(inital_img, bird, cv.TM_CCOEFF_NORMED)

#get the best match position
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

threshold = 0.4

if (max_val > threshold):
    print('Found it')

    bird_w = bird.shape[1]
    bird_h = bird.shape[0]

    top_left = max_loc
    bottom_right = (max_loc[0]+bird_w, max_loc[1]+bird_h)

    cv.rectangle(inital_img, top_left, bottom_right, 
                                color=(0, 0, 255), thickness=2, lineType=cv.LINE_4)
    cv.imwrite(abspath('.\data\\result.jpg'), inital_img)
else:
    print("Can't find it")


cv.imshow("Result", inital_img)
cv.waitKey(0)
