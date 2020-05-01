import deep_ocr as docr
import cv2
import platform
import imutils

from PIL import Image as Img
from tkinter import Tk, Label, Button, Image, Canvas

class gui:
    def __init__(self, tk):

        self.init_camera_stream()
        self.updateVidSetream()

        self.tk = tk # init tk object for self
        tk.title("OCR Remote") # set gui title

        self.label1 = Label(tk, text="Press the button to do OCR on the video stream") # create and set text for label
        self.label1.pack() # pack label into the GUI

        self.capture_button = Button(tk, text="capture img", command=self.getTextInImage) # create and set the button for capturing
        self.capture_button.pack() # pack button into the GUI

    def init_camera_stream(self):
        if self.running_on_jetson_nano() == True:
            self.vs = cv2.VideoCapture(self.get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)
        else:
            self.vs = cv2.VideoCapture(0)
        return

    def getImage(self):
        ret, im = self.vs.read()
        im = imutils.resize(im, width=400, height=400)  # resize the image
        cv2.destroyAllWindows()
        cv2.imshow("capture", im)
        return Img.fromarray(im)

    def getTextInImage(self):
        img = self.getImage()
        text = docr.ocr_win_cv2(img)
        print(text)
        return

    def updateVidSetream(self):
        ret, im = self.vs.read()
        cv2.imshow("Camera Feed", im)
        return

    def running_on_jetson_nano(self):
        return platform.machine() == "aarch64"

    def get_jetson_gstreamer_source(self, capture_width=1280, capture_height=720, display_width=640, display_height=480,
                                    framerate=60, flip_method=2):
        """
        Return an OpenCV-compatible video source description that uses gstreamer to capture video from the camera on a Jetson Nano
        """
        return (
                f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
                f'width=(int){capture_width}, height=(int){capture_height}, ' +
                f'format=(string)NV12, framerate=(fraction){framerate}/1 ! ' +
                f'nvvidconv flip-method={flip_method} ! ' +
                f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! ' +
                'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
        )

root = Tk()
GUI = gui(root)

while 1:
    GUI.updateVidSetream()
    root.update_idletasks()
    root.update()