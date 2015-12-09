from image_segmentation import ImageSegmenter
import image_util
from PIL import Image

numSegments = 5
maxIters = 1

def main():
    segmenter = ImageSegmenter(numSegments,maxIters)
    segmenter.set_weights({"Intensity" : 50, "x": 5, "y": 5, "R":0, "G":0, "B":0})
    im = Image.open("../CS221/stop_1323804419.avi_image33.png")
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
        x_vals = [elem[0] for elem in pixelsInCluster]
        y_vals = [elem[1] for elem in pixelsInCluster]
        im2 = Image.open("../CS221/stop_1323804419.avi_image33.png")
        pixels = im2.load() 
        
        updatedGrid = image_util.isolatePixelsToImage(pixels,pixelsInCluster,min(x_vals),max(x_vals),min(y_vals),max(y_vals),name)
            
     
def get_pixels(im):
    return [[im.load()[x, y] for y in range(im.size[1])]for x in range(im.size[0])]

if __name__ == "__main__":
    main()
