import numpy as np
import cv2

class Yolo_Detection(object):

    # singleton
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Yolo_Detection(), cls).__new__(cls)
        return cls.instance

    # the function that detects the objects
    def apply_detection(video_name: str):

        # initialize the YOLO detector
        net = cv2.dnn.readNet('/mnt/d/POLITEHNICA/AUTOMATICA SI CALCULATOARE/ANUL 2/SEMESTRUL 2/HYPERFY/yolov3.weights', 
                                '/mnt/d/POLITEHNICA/AUTOMATICA SI CALCULATOARE/ANUL 2/SEMESTRUL 2/HYPERFY/yolov3.cfg')
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


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
            width = 640
            height = 480
            dim = (width, height)
            frame = cv2.resize(frame, dim)

            #   preprocess image using blob, resize it
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, dim, swapRB = True, crop = False)
            
            # detect objects
            net.setInput(blob)
            outs = net.forward(output_layers)

            # creating the boxes
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # detect the object
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # creat the rectangle
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            font = cv2.FONT_HERSHEY_PLAIN
            
            # coloured boxes
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    
            # output video and display the frame
            # out.write(frame.astype('uint8'))
            cv2.imshow('Yolo_Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # close everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        pass
