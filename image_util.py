import sys
from PIL import Image

"""
General functions for image processing
"""

def isolatePixelsToImage(pixelGrid,pixelsToKeep,min_x,max_x,min_y,max_y,name):
    img = Image.new( 'RGB', (max_x-min_x + 1,max_y - min_y + 1), "black")
    pixels = img.load()
    for elem in pixelsToKeep:
        i,j = elem
        pixels[i-min_x,j-min_y] = pixelGrid[i,j]
    img.save(name)

