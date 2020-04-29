from PIL import Image
from tesserocr import PyTessBaseAPI

def toBW(img):
    column = Image.open(img)
    gray = column.convert('L')
    bw = gray.point(lambda x: 0 if x < 200 else 255, '1')
    bw.save("bw_im.jpg")
    return "bw_im.jpg"

def ocr(img):
    with PyTessBaseAPI () as api:
        api.SetImageFile(img)
        text = api.GetUTF8Text()
        return text
print(ocr('testimg.jpg'))
