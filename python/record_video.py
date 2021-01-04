
# import the opencv library 
import cv2 as cv
  
  
# define a video capture object 
vid = cv.VideoCapture(0) 
vid.set(cv.CAP_PROP_EXPOSURE, 10)
assert vid.set(cv.CAP_PROP_FRAME_WIDTH, 640)
assert vid.set(cv.CAP_PROP_FRAME_HEIGHT, 360)

num = 0
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Display the resulting frame 
    cv.imshow('frame', frame) 

    if num%30 == 0:
        cv.imwrite(f"images/{num}.png", frame)
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv.waitKey(1) & 0xFF == ord('q'): 
        break

    num += 1
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv.destroyAllWindows() 
