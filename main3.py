from image_segmentation import ImageSegmenter
import image_util
from PIL import Image

numSegments = 50
maxIters = 5

def main():
    segmenter = ImageSegmenter(numSegments,maxIters)
    im = Image.open("../CS221/stop_1323804701.avi_image4.png")
    pix = get_pixels(im)

    segmented = segmenter.segment(get_pixels(im))
    centroids, assignments, oldcost = segmented
    print len(assignments)
    for i in range(0,numSegments):
        name = "temp" + str(i) + ".png"
        pixelsInCluster = []
        for j in range(0, len(assignments)):
            if assignments[j] == i:
                y = j % im.size[1]
                x = (j - y)/im.size[1]      
                pixelsInCluster.append((x,y))
        im2 = Image.open("../CS221/stop_1323804701.avi_image4.png")
        pixels = im2.load() 
        updatedGrid = image_util.isolatePixels(pixels,pixelsInCluster,im2.size[0],im2.size[1])
        for i in range(0,im2.size[0]):
            for j in range(0,im2.size[1]):
                pixels[i,j] = updatedGrid[i,j]
        im2.save(name)
            
     
def get_pixels(im):
    return [[im.load()[x, y] for y in range(im.size[1])]for x in range(im.size[0])]

if __name__ == "__main__":
    main()
