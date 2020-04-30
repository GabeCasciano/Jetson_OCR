from PIL import Image
from tesserocr import PyTessBaseAPI

def ocr(img):
    with PyTessBaseAPI () as api:
        api.SetImageFile(img)
        text = api.GetUTF8Text()
        return text
print(ocr('testimg.jpg'))
