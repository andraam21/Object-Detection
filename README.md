# Object Detection

## Description

This project detects objects and people from an input video or a live stream.

## Structure
There are two singleton classes that contain functions in order to apply a certain detection (Yolo, Hog). Depending on the performance, each filter will find multiple objects in the input, and a rectangle will appear around the target. The classes could be initialized as interfaces, but I have chosen a singleton type because it is easier to extend this pattern (for example, the frame number that is shown can be stored to count how many times the function has been applied).

In the main body, the video is captured, and every frame is processed. If a live stream from the personal camera is desired, the function from the cv2 package can capture from the standard input (0) during the capture of the frames. (In this example, a video sample was used.)

To synchronize the two detections (based on performance, one filter can be faster than the other), the waitKey function is used in each filter's body. Additionally, the detection is applied to each frame simultaneously. Initially, I tried to start two threads (one of them to be a daemon and run in the background) and synchronize them, but I encountered some problems regarding the cv2 image pop-up. In this case, synchronization could have been achieved with a mathematical formula depending on how quickly the object is detected.

## How to run
 In order to run the project use the command: 
 ```
 make
 ```