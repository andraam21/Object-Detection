import yolo as y
import hog as h
import logging
import threading
import time

# apply the first detection
def first_thread():
    print("First thread")
    yolo.apply_detection('people.mp4')

# apply the second detection
def second_thread():
    print("Second thread")
    hog.apply_detection('people.mp4')

# starts the two parallel threads
if __name__ == '__main__':
    yolo = y.Yolo_Detection
    hog = h.Hog_Detection
    threading.Thread(target = first_thread).start()
    threading.Thread(target = second_thread).start()
