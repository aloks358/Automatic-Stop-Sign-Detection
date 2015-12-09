from PIL import Image
import image_util
TEST_PATH = "test.png"

def convertImage():
    im = Image.open(TEST_PATH)
    pixelRawGrid = im.load()
    pixelSet = [(0,0),(1,0),(2,0),(3,0)]
    updatedPixelGrid = image_util.isolatePixels(pixelRawGrid,pixelSet,im.size[0],im.size[1])
    for i in range(0,im.size[0]):
        for j in range(0,im.size[1]):
            pixelRawGrid[i,j] = updatedPixelGrid[i,j]
    im.save("test2.png")    

def createImage():
    img = Image.new( 'RGB', (5,5), "blue")
    img.save(TEST_PATH)

createImage()
convertImage()
