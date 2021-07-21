import cv2
import numpy as np
import time

# capture frames from a camera with device index=0
cap = cv2.VideoCapture(
    "https://deliverys3.joada.net/contents/encodings/live/f154fbd1-742e-4ed5-3335-3130-6d61-63-be54-7f8d574cdffed/master.m3u8")


# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

whT = 320
confTreshold = 0.5
masTreshold = 0.3
count = 0

classesFile = 'coco.names'
# classNames = ["bicycle", "car", "motorbike", "bus", "truck", "train"]
classNames = []

with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# print(classNames)
# print(len(classNames))

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(p_outputs, p_frame):
    hT, wT, cT = p_frame.shape
    bbox = []
    classIds = []
    confs = []

    for output in p_outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confTreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    # print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confTreshold, masTreshold)
    print(indices)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(p_frame, (x, y), (x + w, y + h), (0, 191, 255), 2)
        cv2.putText(p_frame, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


# loop runs if capturing has been initialized
while 1:

    # reads frame from a camera
    ret, frame = cap.read()

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), [0, 0, 0], True, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()

    # print(layerNames)
    # print(net.getUnconnectedOutLayers())

    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)

    outputs = net.forward(outputNames)

    """
    print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2].shape)
    """
    # print(outputs[0][0])

    findObjects(outputs, frame)


    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()

    # Calculating the fps

    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)

    # putting the FPS count on the frame
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Camera', frame)


    # Wait for 25ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera from video capture
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
