import sys
from PIL import Image

def isolatePixels(pixelGrid,pixelsToKeep,dim_x,dim_y):
	for i in range(0,dim_x):
		for j in range(0,dim_y):
			if (i,j) not in pixelsToKeep:
				pixelGrid[i,j] = (0,0,0)
	return pixelGrid
