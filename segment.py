"""
This module performs image segmentation.
"""

import os
import random, copy, math, sys
from PIL import Image
from util import *
import image_util

"""
This class defines an image segmenter.
"""

class ImageSegmenter(object):
    def __init__(self, numSegments, maxIter):
        self.num_segments = numSegments
        self.max_iter = maxIter
        self.features = ["Intensity", "x", "y"]
        self.feature_weights = {"Intensity" : 5, "x": 6, "y": 6, "R": 1, "G": 1, "B": 1}

    """
    Allows adjusting of weights to features when comparing distances.
    """
    def set_weights(self, new_weights):
        if new_weights.keys() != self.feature_weights.keys():
            raise ValueError("The weights must correspond to the features.")
        self.feature_weights = new_weights

    """
    Takes the a 2D array of RGB tuples , and extracts feature
    information for the pixels into a list.
    """
    def convert_image_to_pixels(self, image):
        pixels = []
        for x in range(len(image)):
            for y in range(len(image[0])):
                pixels.append({
                    "Intensity" : self.intensity_calc(image[x][y]),
                    "x": x,
                    "y": y,
					"R": image[x][y][0],
					"G": image[x][y][1],
					"B": image[x][y][2]})
        return pixels

    """
    Takes a pixel (a RGB tuple) and returns its intensity (grayscale value)
    """
    def intensity_calc(self, pixel):
        intensity = pixel[0]*0.2989 + pixel[1]*0.5870 +pixel[2]*0.1140
        return intensity

    """
    Actually performs image segmentation on an image represented by a list of RGB tuples.
    """
    def segment(self, image):
        pixels = self.convert_image_to_pixels(image)
        def calc_distance(pixel, center):
            return math.sqrt(sum([((pixel[elem]-center[elem])**2)*self.feature_weights[elem] for elem in pixel]))
        n = len(pixels)
        centroids = [pixels[random.randint(0, n - 1)] for k in range(self.num_segments)]
        assignments = [None]*n
        old_cost = None
        for t in range(self.max_iter):
            print t, old_cost
            total_cost = 0
            for i, pixel in enumerate(pixels):
                min_cost = None
                assignments[i] = 0
                for k in range(self.num_segments):
                    cost = calc_distance(pixel, centroids[k])
                    if min_cost == None or cost < min_cost:
                        min_cost = cost
                        assignments[i] = k
                total_cost += min_cost

            if total_cost == old_cost:
                break
            old_cost = total_cost

            for k in range(self.num_segments):
                pixels_cluster = [pixel for i, pixel in enumerate(pixels) if assignments[i] == k]
                if len(pixels_cluster) > 0:
                    first_pixel = copy.deepcopy(pixels_cluster[0])
                    for i in range(1, len(pixels_cluster)):
                        increment(first_pixel, 1, pixels_cluster[i])
                    average = {}
                    for elem in first_pixel:
                        average[elem] = float(first_pixel[elem])/len(pixels_cluster)
                        centroids[k] = average
        return centroids, assignments, old_cost

"""
Takes a path to an image, sets the number of segments and iteration parameters for
K-means, and creates the segments.
"""
def main(path):
    numSegments = 25
    maxIters = 10
    segmenter = ImageSegmenter(numSegments,maxIters)
    segmenter.set_weights({"Intensity" : 50, "x": 5, "y": 5, "R":0, "G":0, "B":0})
    im = Image.open(path)
    pix = get_pixels(im)

    segmented = segmenter.segment(get_pixels(im))
    centroids, assignments, oldcost = segmented
    print len(assignments)
    for i in range(0,numSegments):
        name = os.path.join("RESULTS", path.replace('/', '__')) +  "temp" + str(i) + ".png"
        pixelsInCluster = []
        for j in range(0, len(assignments)):
            if assignments[j] == i:
                y = j % im.size[1]
                x = (j - y)/im.size[1]
                pixelsInCluster.append((x,y))
        x_vals = [elem[0] for elem in pixelsInCluster]
        y_vals = [elem[1] for elem in pixelsInCluster]
        im2 = Image.open(path)
        pixels = im2.load()

        updatedGrid = image_util.isolatePixelsToImage(pixels,pixelsInCluster,min(x_vals),max(x_vals),min(y_vals),max(y_vals),name)

"""
Given an image, returns a 2D array of RGB tuples.
"""
def get_pixels(im):
    return [[im.load()[x, y] for y in range(im.size[1])]for x in range(im.size[0])]

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print "usage: segment.py <image>"
    else:
        main(sys.argv[1])
