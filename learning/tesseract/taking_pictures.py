import cv2
import platform
import imutils
import sys
from PIL import Image as Img
import time

vs = None

def get_jetson_gstreamer_source(capture_width=1280, capture_height=720, display_width=640, display_height=480,
                                framerate=30, flip_method=2):
    """
    Return an OpenCV-compatible video source description that uses gstreamer to capture video from the camera on a Jetson Nano
    """
    return (
            f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
            f'width=(int){1280}, height=(int){720}, ' +
            f'format=(string)NV12, framerate=(fraction){30}/1 ! ' +
            f'nvvidconv flip-method={2} ! ' +
            f'video/x-raw, width=(int){640}, height=(int){480}, format=(string)BGRx ! ' +
            'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
    )

def running_on_jetson_nano():
    return platform.machine() == "aarch64"

def init_camera_stream(vs):
    # if we are running on jetson
    if running_on_jetson_nano() == True:
        # generate jetson camera feed
        vs = cv2.VideoCapture(get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)
    else:
        # generate normal camera feed
        vs = cv2.VideoCapture(0)

    return vs

vs = init_camera_stream(vs);

for i in range(0, 10):
    ret, img = vs.read()
    cv2.imwrite(f"./pictures/picture_{i}.jpg", img)
    time.sleep(0.2)