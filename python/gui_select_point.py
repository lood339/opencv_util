import cv2 as cv

import numpy as np

im = cv.imread('/Users/jimmy/Desktop/house.png', 1)
points = []

vis = im.copy()
# mouse callback function
def draw_circle(event,x,y,flags,param):

    vis = im.copy()
    # add a point
    if event == cv.EVENT_LBUTTONDOWN:
        points.append([x, y])
    # delete a closest point in the
    elif event == cv.EVENT_RBUTTONDOWN:
        pass
    print(len(points))

    for p in points:
        cv.circle(vis, (p[0], [1]), 5, (255, 0, 0), 2)


# Create a black image, a window and bind the function to window
cv.namedWindow('selected points')
cv.setMouseCallback('selected points', draw_circle)

while(1):
    cv.imshow('selected points', vis)
    if cv.waitKey(20) & 0xFF == 27:
        break
cv.destroyAllWindows()