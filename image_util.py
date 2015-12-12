import sys
from PIL import Image

"""
General functions for image processing
"""


"""
Given a full image and the coordinates of its 
segment's corners, generates and saves a new 
image of the segment alone
"""
def isolatePixelsToImage(pixelGrid,pixelsToKeep,min_x,max_x,min_y,max_y,name):
    img = Image.new( 'RGB', (max_x-min_x + 1,max_y - min_y + 1), "black")
    pixels = img.load()
    for elem in pixelsToKeep:
        i,j = elem
        pixels[i-min_x,j-min_y] = pixelGrid[i,j]
    img.save(name)


"""
Given an image, returns a 2D array of RGB tuples.
"""
def get_pixels(im):
    return [[im.load()[x, y] for y in range(im.size[1])]for x in range(im.size[0])]
