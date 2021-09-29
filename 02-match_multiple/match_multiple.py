import cv2 as cv
import numpy as np
from os.path import abspath


print_path = abspath('.\data\\print2.png')
bird_path = abspath('.\data\\flapbird_bird.PNG')
tube_bottom_path = abspath('.\data\\tube_bottom.png')
tube_top_path = abspath('.\data\\tube_top.png')

inital_img = cv.imread(print_path, cv.IMREAD_REDUCED_COLOR_2)
bird = cv.imread(bird_path, cv.IMREAD_REDUCED_COLOR_2)
tube_bottom = cv.imread(tube_bottom_path, cv.IMREAD_REDUCED_COLOR_2)
tube_top = cv.imread(tube_top_path, cv.IMREAD_REDUCED_COLOR_2)

result_bird = cv.matchTemplate(inital_img, bird, cv.TM_CCOEFF_NORMED)
result_tube_bottom = cv.matchTemplate(inital_img, tube_bottom, cv.TM_CCOEFF_NORMED)
result_tube_top = cv.matchTemplate(inital_img, tube_top, cv.TM_CCOEFF_NORMED)

#get the best match position
#min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result_bird)
minmax_bird = cv.minMaxLoc(result_bird)
minmax_tube_bottom = cv.minMaxLoc(result_tube_bottom)
minmax_tube_top = cv.minMaxLoc(result_tube_top)

threshold = 0.4
threshold_tubes = 0.96

tube_location_bottom = np.where(result_tube_bottom >= threshold_tubes)
tube_location_top    = np.where(result_tube_top    >= threshold_tubes)

#pack the indexes in tuples
tube_location_bottom = list(zip(*tube_location_bottom[::-1]))
tube_location_top    = list(zip(*tube_location_top[::-1]))


if (minmax_bird[1] > threshold):
    print('Found it')

    #bird dimensions
    bird_w = bird.shape[1]
    bird_h = bird.shape[0]

    #tube bottom dimension
    tube_bottom_w = tube_bottom.shape[1]
    tube_bottom_h = tube_bottom.shape[0]

    #tube top dimension
    tube_top_w = tube_top.shape[1]
    tube_top_h = tube_top.shape[0]

    ##top left and bottom right points for the rectangle || bird position
    top_left_bird = minmax_bird[3]
    bottom_right_bird = (minmax_bird[3][0]+bird_w, minmax_bird[3][1]+bird_h)
    
    ##top left and bottom right points for the rectangle || tube_bottom position
    top_left_tube_bottom = minmax_tube_bottom[3]
    bottom_right_tube_bottom = (minmax_tube_bottom[3][0]+tube_bottom_w, minmax_tube_bottom[3][1]+tube_bottom_h)

    ##top left and bottom right points for the rectangle || tube_top position
    top_left_tube_top = minmax_tube_top[3]
    bottom_right_tube_top = (minmax_tube_top[3][0]+tube_top_w, minmax_tube_top[3][1]+tube_top_h)
    

    #Draw rectangle arrond the bird
    cv.rectangle(inital_img, top_left_bird, bottom_right_bird, 
                                color=(0, 0, 255), thickness=2, lineType=cv.LINE_4)

    #Draw rectangles arrond the tubes on the top of the screen
    for loc in tube_location_bottom:
        top_left_tube_bottom = loc
        bottom_right_tube_bottom = (top_left_tube_bottom[0] + tube_bottom_w, top_left_tube_bottom[1] + tube_bottom_h )
        cv.rectangle(inital_img, top_left_tube_bottom, bottom_right_tube_bottom, 
                                color=(255, 255, 0), thickness=2, lineType=cv.LINE_4)

    #Draw rectangles arrond the tubes on the top of the screen
    for loc in tube_location_top:
        top_left_tube_top = loc
        bottom_right_tube_top = (top_left_tube_top[0] + tube_top_w, top_left_tube_top[1] + tube_top_h )
        cv.rectangle(inital_img, top_left_tube_top, bottom_right_tube_top, 
                                color=(255, 255, 0), thickness=2, lineType=cv.LINE_4)
    
    
    cv.imwrite(abspath('.\data\\result.jpg'), inital_img)
else:
    print("I couldnt find the bird")


cv.imshow("Result", inital_img)
cv.waitKey(0)


print(tube_location_top)