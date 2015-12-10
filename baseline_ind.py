import csv
import sys, os
import util

from PIL import Image


DATA_PATH = "../LISA_TS"
LABEL_FILE = "../CS221/allAnnotations.csv"
NUM_ITERATIONS = 10

def featureExtractor(imagePath):
    #thresholds
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
            if b < bt and g < gt and r > rt:
                featureVec[(r,g,b)] = 1

    return featureVec

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

def filterTrainExamples(trainExamples):
    temp = []
    for i in range(0,len(trainExamples)):
        tEx = trainExamples[i]
        if os.path.isfile(DATA_PATH + "/" + tEx[0]) == True:
            temp.append(tEx)
    return temp

def main():
    trainExamples = get_image_labels()
    train = filterTrainExamples(trainExamples)
    testExamples = []
    
    print trainExamples
    print util.SGD(train[0:100], testExamples, featureExtractor, debug = True)
    print "test"

if __name__ == "__main__":
    main()
