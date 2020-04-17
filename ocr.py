import numpy as np
import cv2
import imutils
from imutils.video import VideoStream

black = (0,0,0)
light_black = (0, 10, 10)

kernel = np.ones((2,2), np.uint8)

pi_cam_parameters = (
            f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
            f'width=(int){1280}, height=(int){720}, ' +
            f'format=(string)NV12, framerate=(fraction){60}/1 ! ' +
            f'nvvidconv flip-method={2} ! ' +
            f'video/x-raw, width=(int){640}, height=(int){480}, format=(string)BGRx ! ' +
            'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
            )

def main_loop():
    vs = cv2.VideoCapture(pi_cam_parameters, cv2.CAP_GSTREAMER)

    while True:
        ret, frame = vs.read()
        frame = imutils.resize(frame, width=600, height=400)

        blurred = cv2.GaussianBlur(frame, (3,3), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        masked = cv2.inRange(hsv, light_black, black)
        dilated = cv2.dilate(masked, kernel, iterations=1)
        erroded = cv2.erode(dilated, kernel, iterations=1)

        cv2.imshow("Origianl", frame)
        cv2.imshow("Masked", masked)
        cv2.imshow("filtered", erroded)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Done")
            break
    vs.stop()
    vs.release()
    cv2.destroyAllWindows()
    return

main_loop()