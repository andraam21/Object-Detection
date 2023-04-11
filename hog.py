import numpy as np
import cv2

class Hog_Detection(object):

    # singleton
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Hog_Detection(), cls).__new__(cls)
        return cls.instance

    # the function that detects the objects
    def apply_detection(video_name: str):

        # initialize the HOG detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        cv2.startWindowThread()

        # capture the video
        cap = cv2.VideoCapture(video_name)

        # save the new video
        # out = cv2.VideoWriter(
        #     'output.mp4',
        #     cv2.VideoWriter_fourcc(*'MJPG'),
        #     15.,
        #     (640,480))

        while True:
            # take every frame
            ret, frame = cap.read()

            # resizing and transform to gray to get the detection faster
            dim = (640, 480)
            frame = cv2.resize(frame, dim)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # detect people in the image
            boxes, weights = hog.detectMultiScale(frame, winStride=(8,8))
            boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
            
            # coloured boxes
            for i in range(len(boxes)):
                if len(boxes[i]) == 4:
                    x, y, w, h =  boxes[i]
                    cv2.rectangle(frame, (x, y), (w, h), (0, 255, 255), 2)

            # output video and display the frame
            # out.write(frame.astype('uint8'))
            cv2.imshow('Hog_Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # close everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        pass
