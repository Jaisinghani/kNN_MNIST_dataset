from collections import defaultdict
import matplotlib.pyplot as plt
import struct
from numpy import *
import numpy as np
import itertools
import datetime

def parse_labels(fileName):
    """Parse labels from the binary file."""
    filePath = "./dataset/" + fileName
    with open(filePath, "rb") as binary_file:
        data = binary_file.read()
    magic, n = struct.unpack_from('>2i', data)
    assert magic == 2049, "Wrong magic number: %d" % magic

    labels = struct.unpack_from('>%dB' % n, data, offset=8)
    return labels


def parse_images(fileName):
    """Parse images from the binary file."""
    filePath= "./dataset/"+fileName
    with open(filePath, "rb") as binary_file:
        data = binary_file.read()

    magic, n, rows, cols = struct.unpack_from('>4i', data)
    assert magic == 2051, "Wrong magic number: %d" % magic

    num_pixels = n * rows * cols
    pixels = struct.unpack_from('>%dB' % num_pixels, data, offset=16)


    pixels = asarray(pixels, dtype=np.float)


    images = pixels.reshape((n, cols, rows))
    return images


print("loading training images")
trainImagesSet = parse_images("train-images-idx3-ubyte")
print("size of train image set---->",size(trainImagesSet))
print("shape of train image set---->",shape(trainImagesSet))


print("loading test images")
testImagesSet = parse_images("t10k-images-idx3-ubyte")
print("size of test image set---->",size(testImagesSet))
print("shape of test image set---->",shape(testImagesSet))


print("loading training labels")
trainLabelsSet = parse_labels("train-labels-idx1-ubyte")
print("size of train image set---->",size(trainLabelsSet))
print("shape of train image set---->",shape(trainLabelsSet))


print("loading test labels")
testLabelsSet = parse_labels("t10k-labels-idx1-ubyte")
print("size of test image set---->",size(testLabelsSet))
print("shape of test image set---->",shape(testLabelsSet))


def calEuclideanDistance(point1, point2):
    distance=  np.sqrt(np.sum((point1-point2)**2))
    return distance


def get_majority(votes):
    counter = defaultdict(int)
    for vote in votes:
        counter[vote] += 1

    majority_count = max(counter.values())
    for key, value in counter.items():
        if value == majority_count:
            return key
dataset = []
for i in range(20):
    dataset.append((trainImagesSet[i, :, :], trainLabelsSet[i],i))


#distance calculations
imagedistances = []
for index, value in enumerate(dataset):
    distances =[]
    for y in range(index + 1, len(dataset)):
        distance = calEuclideanDistance(value[0], dataset[y][0])
        print("label of value :",value[1] )
        print("label of data :", dataset[y][1])
        distances.append((distance, y))
    imagedistances.append((index, distances))
print("image distances :", imagedistances)

# distances = []
#
#
# for index, value in enumerate(dataset):
#     for i in range(index+1, len(dataset)):
#         distance = calEuclideanDistance(value[0], dataset[i][0])
#         print("distance between ", i, " and ", index, "is", distance)


#Copy of training image set
# copyOfTrainingDataSet = dataset[:]
# np.random.shuffle(copyOfTrainingDataSet)
#
#
# #Creating 10-folds
# foldedArray = []
# start = 0
# end = 6000
# i=0
# for i in range(10):
#     arr = copyOfTrainingDataSet[start:end]
#     start+= 6000
#     end+=6000
#     foldedArray.append(arr)


