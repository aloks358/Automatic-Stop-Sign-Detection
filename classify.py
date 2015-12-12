"""
Performs the classification for an entirely new image by segmenting it and seeing if any of the segments are stop signs.
"""

import sys, os
import train, util
import segment_util as seg_util

SEGMENTS_PATH = "RESULTS"

def main():

    path = ""
    image_name = ""
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if not os.path.exists(path):
            print "The path provided does not exist."
            return
        directories = path.split('/')
        file_name = directories[len(directories) - 1]
    else:
        print "Please supply a path to an image."
        return

    os.system("python segment.py " + path)
    segments = []
    for f in os.listdir(SEGMENTS_PATH):
        if 'temp' in f and image_name in f: # Ways to identify segments of the given path
             segments.append(os.path.join(SEGMENTS_PATH,f))

    f = open('weights.out')   # Read in weights
    weights = eval(f.readline())

    stop_sign_flag = False
    for segment in segments:
        score = weights * seg_util.segmentFeatureExtractor(segment)

        if score >= 0:  # Stop sign found
            stop_sign_flag = True
            break

    if stop_sign_flag:
        print "Stop sign detected!"
    else:
        print "No stop sign detected"


if __name__ == "__main__":
    main()
