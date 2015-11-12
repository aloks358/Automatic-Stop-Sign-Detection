import sys
from PIL import Image

def isolatePixels(pixelGrid,pixelsToKeep,dim_x,dim_y):
    img = Image.new( 'RGB', (dim_x,dim_y), "black")
    pixels = img.load()
    for elem in pixelsToKeep:
        i,j = elem
        pixels[i,j] = pixelGrid[i,j]
    return pixels

