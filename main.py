import yolo as y
import hog as h
import threading
import time
import cv2

# capture the video
video_name = 'people.mp4'
cap = cv2.VideoCapture(video_name)

# applying simultaneously the two filters
while True:
    ret, frame = cap.read()
    y.Yolo_Detection.apply_detection(frame)
    h.Hog_Detection.apply_detection(frame)

# releasing the video and closing the tabs
cap.release()
cv.destroyAllWindows()
cv.waitKey(0)
