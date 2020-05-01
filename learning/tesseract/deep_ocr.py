from PIL import Image
from tesserocr import PyTessBaseAPI

#Runs on windows, from image
def ocr_win_cv2(img):
    with PyTessBaseAPI(path="C:/Users/Gabe/PycharmProjects/Python_venv/tessdata-master/tessdata-master") as api:
        api.SetImage(img)
        text = api.GetUTF8Text()
        return text

#Runs on windows, from file
def ocr_win(img):
    with PyTessBaseAPI(path="C:/Users/Gabe/PycharmProjects/Python_venv/tessdata-master/tessdata-master") as api:
        api.SetImageFile(img)
        text = api.GetUTF8Text()
        return text

#Runs on jetson, from file
def ocr_jet(img):
    with PyTessBaseAPI() as api:
        api.SetImageFile(img)
        text = api.GetUTF8Text()
        return text

#Runs on jetson, from image
def ocr_jet_cv2(img):
    with PyTessBaseAPI() as api:
        api.SetImage(img)
        text = api.GetUTF8Text()
        return text

