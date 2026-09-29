import cv2 as cv
import numpy as np

print("OpenCV: ", cv.__version__)
img = np.zeros((120, 400, 3), dtype=np.uint8)

cv.putText(img, "OPENCV OK", (10,30), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
cv.imshow("hello", img); 
cv.waitKey(0)