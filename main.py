import cv2
import numpy as np
import time
from prometheus_client import start_http_server, Gauge, Counter

g = Counter('vehicles_count', 'Number of vehicles', ["location", "direction"])
current_location = "Viaduc_de_Millau"

cap = cv2.VideoCapture(
    "https://deliverys2.joada.net/contents/encodings/live/f154fbd1-742e-4ed5-3335-3130-6d61-63-be54-7f8d574cdffed/mpd.mpd")

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

# Substraction
sub = cv2.bgsegm.createBackgroundSubtractorMOG()

min_w_rect = 36
min_h_rect = 36

detect = []
offeset = 3

counter_right = 0
counter_left = 0


def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

start_http_server(8000)

while 1:
    ret, frame = cap.read()

    try:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 5)
        img_sub = sub.apply(blur)
        dilat = cv2.dilate(img_sub, np.ones((5, 5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
        dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
        counter, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        continue

    # cv2.imshow('Detecter', dilatada)

    # Right line
    cv2.line(frame, (610, 550), (793, 567), (255, 127, 0), 3)

    # Left line
    cv2.line(frame, (280, 520), (536, 525), (0, 127, 255), 3)

    # Generic Line
    # cv2.line(frame, (280, 550), (793, 550), (255, 127, 0), 3)

    for (i, c) in enumerate(counter):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_w_rect) and (w >= min_h_rect)
        if not validate_counter:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

    for (x, y) in detect:
        if y > (550 - offeset) and y < (550 + offeset) and x > 280 and x < 525:
            counter_left += 1
            g.labels(location=current_location, direction='left').inc()
        if y > (550 - offeset) and y < (550 + offeset) and x > 610 and x < 793:
            counter_right += 1
            g.labels(location=current_location, direction='right').inc()

        detect.remove((x, y))

    cv2.putText(frame, "Left: " + str(counter_left), (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 100, 100], 2)
    cv2.putText(frame, "Right: " + str(counter_right), (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 100, 100], 2)

    cv2.rectangle(frame, (0, 350), (950, 720), (255, 255, 0), 2)

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

    cv2.imshow('Camera', frame)

    # Wait for 25ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
