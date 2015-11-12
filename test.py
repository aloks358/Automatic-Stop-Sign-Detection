import image_util
from wand.image import Image
import array
from wand.color import Color

TEST_PATH = "test.png"


x = 5
y = 5

def convert_image_to_pixels(image):
	pixels = []
	for i in range(len(image)/y):
		row = []
		for j in range(x):
			row.append(image[i*y+j])	
			#row.append({"Intensity" : intensity_calc(image[i*y + j]), "x": j, "y": i})
		pixels.append(row)
	return pixels

def featureExtractor(imagePath):
    rawpixels = []
    im  = Image(filename = imagePath)
    w, h = im.width, im.height
    blob = im.make_blob(format='RGB')
    for cursor in range(0, w*h*3,3):
        rawpixels.append((blob[cursor], blob[cursor+1], blob[cursor+2]))
    return rawpixels

def convertImage():
    pixelGrid = convert_image_to_pixels(featureExtractor(TEST_PATH))
    pixelSet = {(0,0),(1,0),(1,1),(0,1)}
    updatedPixelGrid = image_util.isolatePixels(pixelGrid,pixelSet,x,y)
    return updatedPixelGrid

def flatten(grid,x,y):
	vec = []
	for i in range(0,x):
		for j in range(0,y):
			vec.append(grid[i][j][0])
			vec.append(grid[i][j][1])
			vec.append(grid[i][j][2])
	barr = bytearray(vec)
	return str(barr)

def test():
	with Image(width=5,height=5) as img:
		with Image(width=img.width, height=img.height, background=Color("white")) as bg:
			bg.composite(img,0,0)
			bg.save(filename="test.png")
	upGrid = convertImage()
	with Image(blob=flatten(upGrid,x,y),format='RGB',width=5,height=5) as img1:
		img1.save('my.png')

test()
