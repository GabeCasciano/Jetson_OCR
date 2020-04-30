from PIL import Image
from tesserocr import PyTessBaseAPI

#Runs on windows
def ocr_win(img):
    with PyTessBaseAPI(path="C:/Users/Gabe/PycharmProjects/Python_venv/tessdata-master/tessdata-master") as api:
        api.SetImageFile(img)
        text = api.GetUTF8Text()
        return text

#Runs on jetson
def ocr_jet(img):
    with PyTessBaseAPI() as api:
        api.SetImageFile(img)
        text = api.GetUTF8Text()
        return text

print(ocr_win('ocrimage.jpg'))
