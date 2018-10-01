from collections import defaultdict
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import struct
from numpy import *

print("Hello")


def parse_labels(fileName):
    """Parse labels from the binary file."""
    filePath = "./dataset/" + fileName
    with open(filePath, "rb") as binary_file:
        data = binary_file.read()
        # We're going to use the Python 'struct' package.
        # This is an incredibly nice package which allows us to specify the format
        # our data is in, and then automatically parses the data from the string.
        # Let's start by getting the magic number and the length: the first character
        # represents the endianness of the data (in our case, '>' for big endian), while
        # the next characters are the format string ('2i' for two integers).
    magic, n = struct.unpack_from('>2i', data)
    assert magic == 2049, "Wrong magic number: %d" % magic

    # Next, let's extract the labels.
    labels = struct.unpack_from('>%dB' % n, data, offset=8)
    return labels


def parse_images(fileName):
    """Parse images from the binary file."""
    filePath = "./dataset/" + fileName
    with open(filePath, "rb") as binary_file:
        data = binary_file.read()

        # Parse metadata.
    magic, n, rows, cols = struct.unpack_from('>4i', data)
    assert magic == 2051, "Wrong magic number: %d" % magic

    # Get all the pixel intensity values.
    num_pixels = n * rows * cols
    pixels = struct.unpack_from('>%dB' % num_pixels, data, offset=16)

    # Convert this data to a NumPy array for ease of use.
    pixels = asarray(pixels, dtype=ubyte)

    # Reshape into actual images instead of a 1-D array of pixels.
    images = pixels.reshape((n, cols, rows))
    return images


print("loading training images")
trainImagesSet = parse_images("train-images-idx3-ubyte")
print("size of train image set---->", size(trainImagesSet))
print("shape of train image set---->", shape(trainImagesSet))

print("loading test images")
testImagesSet = parse_images("t10k-images-idx3-ubyte")
print("size of test image set---->", size(testImagesSet))
print("shape of test image set---->", shape(testImagesSet))

print("loading training labels")
trainLabelsSet = parse_labels("train-labels-idx1-ubyte")
print("size of train image set---->", size(trainLabelsSet))
print("shape of train image set---->", shape(trainLabelsSet))

print("loading test labels")
testLabelsSet = parse_labels("t10k-labels-idx1-ubyte")
print("size of test image set---->", size(testLabelsSet))
print("shape of test image set---->", shape(testLabelsSet))


class Knnclassifier:
    def __init__(self, dataset, k):
        self.dataset = dataset
        self.k = k

    def predict(self, point):
        distances = []
        for index, value in enumerate(self.dataset):
            distance = self.distance(value[0], point)
            distances.append((distance, index))

        distances.sort(key=lambda val: val[0])

        indexesOfKImages = []
        for i in range(self.k):
            indexesOfKImages.append(distances[i][1])

        labels = []
        for j in indexesOfKImages:
            labels.append(self.dataset[j][1])

        prediction = self.observe(labels)
        return prediction


class MNISTPredictor(Knnclassifier):

    def distance(self, p1, p2):
        dist = calEuclideanDistance(p1, p2)
        return dist

    def observe(self, values):
        return get_majority(values)


def calEuclideanDistance(point1, point2):
    distance = np.sqrt(np.sum((point1 - point2) ** 2))
    return distance


def get_majority(votes):
    # For convenience, we're going to use a defaultdict.
    # This is just a dictionary where values are initialized to zero
    # if they don't exist.
    counter = defaultdict(int)
    for vote in votes:
        # If this weren't a defaultdict, this would error on new vote values.
        counter[vote] += 1

    # Find out who was the majority.
    majority_count = max(counter.values())
    for key, value in counter.items():
        if value == majority_count:
            return key


def predict_test_set(predictor, test_set):
    """Compute the prediction for every element of the test set."""

    predictions = [predictor.predict(test_set[i]) for i in range(len(test_set))]
    return predictions


def evaluate_prediction(predictions, answers):
    """Compute how many were identical in the answers and predictions,
    and divide this by the number of predictions to get a percentage."""
    correct = sum(asarray(predictions) == asarray(answers))
    total = float(prod(len(answers)))
    return correct / total


# Convert our data set into an easy format to use.
# This is a list of (x, y) pairs. x is an image, y is a label.
indexarr = list(range(0,10000))
print("before shuffle:", indexarr)
np.random.shuffle(indexarr)
print("after shuffle:", indexarr)

dataset = []
for index in range(len(indexarr)):
    dataset.append((trainImagesSet[index, :, :], trainLabelsSet[index]))

copyOfTrainingDataSet = dataset[:]



# Creating 10-folds
foldedArray = []
start = 0
end = 6000
i = 0
for i in range(10):
    arr = copyOfTrainingDataSet[start:end]
    start += 6000
    end += 6000
    foldedArray.append(arr)

#generating test data and training data from 10 folds
fold_10_accuracy_each_k = []
for fold in range(len(foldedArray)):
    training_set = []
    training_set = foldedArray[:]
    test_labels = []
    testing_set = []
    train_set_fold_copy = training_set[fold]
    test_set = train_set_fold_copy[:]
    print("size of test_set---->", len(test_set))

    for i in range(len(test_set)):
        test_labels.append(test_set[i][1])
        testing_set.append(test_set[i][0])

    print("size of test labels :", len(test_labels))

    del training_set[fold]
    print("size of training_set---->", len(training_set))
    fold_training_set = list(itertools.chain.from_iterable(training_set))
    predictor = MNISTPredictor(fold_training_set, 3)
    one_train_predictions = predict_test_set(predictor, testing_set)
    one_train_set_accuracy = evaluate_prediction(one_train_predictions, test_labels)
    fold_10_accuracy_each_k.append(one_train_set_accuracy)
    print("accuracies for one fold 9 training set k =7 : ", one_train_set_accuracy)
average_accuracy = sum(fold_10_accuracy_each_k)/10
print("average_accuracy each k :", average_accuracy)