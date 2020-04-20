import numpy as np
import cv2
import imutils

lower = (0, 0, 0)
upper = (255, 255, 10)

IMAGE = 'char_set_1.jpg' # put the name of your file here

kernel = np.ones((2,2), np.uint8)

def main_loop():

    image = cv2.imread(IMAGE)
    # frame = imutils.resize(image, width=600, height=400)

    # blurred = cv2.GaussianBlur(frame, (3,3), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    masked = cv2.inRange(hsv, lower, upper)
    dilated = cv2.dilate(masked, kernel, iterations=1)
    erroded = cv2.erode(dilated, kernel, iterations=1)

    cv2.imshow("Original", image)
    cv2.imshow("Masked", masked)
    cv2.imshow("filtered", erroded)

    key = cv2.waitKey(1) & 0xFF
    while key != ord("q"):
        key = cv2.waitKey(1) & 0xFF

    print("Done")
    cv2.destroyAllWindows()

    return

main_loop()