from PIL import Image
TEST_PATH = "test.png"

def convertImage():
    im = Image.open(TEST_PATH)
    pixelRawGrid = im.load()
    pixelSet = {(0,0),(1,0),(1,1),(0,1)}
    updatedPixelGrid = image_util.isolatePixels(pixelGrid,pixelSet,x,y)
    return updatedPixelGrid

def test():

test()
