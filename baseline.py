import csv
import sys, os
import util
import random
from PIL import Image

"""
Baseline implementation. The baseline implementation makes every pixel a feature by considering if the rgb values of that pixel satisfy some threshold, and uses a linear classifier.
"""

DATA_PATH = "../LISA_TS"
LABEL_FILE = "../CS221/allAnnotations.csv"
NUM_ITERATIONS = 10

"""
Extracts the pixel features (mapping a pixel's coordinates to whether or not it satisfying some trhreshold indicating red).
"""
def featureExtractor(imagePath):
    # Thresholds
    bt = 10
    gt = 10
    rt = 0

    rawpixels = []
    im = Image.open(DATA_PATH + "/" + imagePath)
    rgb_im = im.convert('RGB')

    featureVec = {}
    for i in range(0, im.size[0]):
        for j in range(0,im.size[1]):
            r, g, b = rgb_im.getpixel((i,j))
            if r > rt and g < gt and b < bt:
                featureVec[(r,g,b)] = 1

    return featureVec

"""
Reads in an Excel spreadsheet and notes down which images
(where a path to an image corresponds to an image) are stop
signs or not as per the information in the spreadsheet.
"""
def get_image_labels():
    label_tuples = []
    with open(LABEL_FILE, 'rb') as labels_file:
        labelreader = csv.reader(labels_file, dialect='excel')
        counter = 0
        for row in labelreader:
            if counter == 0:
                counter = 1
                continue
            line = row[0]
            split_line = line.split(';')
            label = -1
            if split_line[1] == "stop":
                label = 1
            label_tup = (split_line[0], label)
            label_tuples.append(label_tup)

    return label_tuples


"""
Makes sure training examples are files. 
"""
def filterTrainExamples(trainExamples):
    filtered_examples = []
    for i in range(0, len(trainExamples)):
        tEx = trainExamples[i]
        if os.path.isfile(os.path.join(DATA_PATH, tEx[0])):
            filtered_examples.append(tEx)
    return filtered_examples

"""
Gets training examples, randomly selects some of them, and then runs
SGD on them with a test set.
"""
def main():
    trainExamples = get_image_labels()
    train = filterTrainExamples(trainExamples)
    random.shuffle(train)
    train = train[0:500]  
    for elem in train:
        if elem[1] == 1:
            print "YES"
    testExamples = train[len(train)/2:]
    print len(train[:len(train)/2])
    print len(testExamples)
    print util.SGD(train[:len(train)/2], testExamples, featureExtractor, debug = True)

if __name__ == "__main__":
    main()
