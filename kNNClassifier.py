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

    def predictionslidingwindow(self,point):
        distances = []
        for index, value in enumerate(self.dataset):
            train_image = value[0]

            #padded training image to convert from 28*28 to 30*30
            paddedImage = np.pad(train_image, [(1, 1), (1, 1)], mode='constant')

            #slice train image into 9 images to find min distance
            sliding_window_distances = []
            for i in range(3):
                for j in range(3):
                    slicedImage = paddedImage[i:i + 28, j:j + 28]
                    distance = self.distance(slicedImage, point)
                    sliding_window_distances.append(distance)
            distances.append((min(sliding_window_distances), index))
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


def predict_test_set(predictor, test_set):
    """Compute the prediction for every element of the test set."""

    predictions = [predictor.predict(test_set[i]) for i in range(len(test_set))]
    return predictions


def evaluate_prediction(predictions, answers):
    correct = sum(asarray(predictions) == asarray(answers))
    total = float(prod(len(answers)))
    return correct / total


#Shuffling the training Image Set
dataset = []
for i in range(len(trainImagesSet)):
    dataset.append((trainImagesSet[i, :, :], trainLabelsSet[i]))

#Copy of training image set
copyOfTrainingDataSet = dataset[:]
np.random.shuffle(copyOfTrainingDataSet)


#Creating 10-folds
foldedArray = []
start = 0
end = 6000
i=0
for i in range(10):
    arr = copyOfTrainingDataSet[start:end]
    start+= 6000
    end+=6000
    foldedArray.append(arr)

#10-CROSS VALIDATION
k_accuracy =[]
ks = [7,9]
for k in ks:
    print(datetime.datetime.now())
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
        predictor = MNISTPredictor(fold_training_set, k)
        one_train_predictions = predict_test_set(predictor, testing_set)
        one_train_set_accuracy = evaluate_prediction(one_train_predictions, test_labels)
        fold_10_accuracy_each_k.append(one_train_set_accuracy)
        print("accuracies for one fold 9 training set k =7 : ", one_train_set_accuracy)

    average_accuracy = sum(fold_10_accuracy_each_k)/10
    print("average_accuracy each k :", average_accuracy)
    k_accuracy.append(average_accuracy)
print("all k accuracy :", k_accuracy)
optimal_k = k_accuracy.index(max(k_accuracy))+1
print("optimal k :", optimal_k)

#running knn for optimal K
predictor = MNISTPredictor(copyOfTrainingDataSet, optimal_k)
optimal_k_predictions = predict_test_set(predictor, testImagesSet)


#Accuracy and error calculations
optimal_k_accuracy = evaluate_prediction(optimal_k_predictions, testLabelsSet)
optimal_k_classification_accuracy  = optimal_k_accuracy*100
print(" Classification accuracy for optimal K :"+ optimal_k_classification_accuracy)
optimal_k_classification_error = (1-optimal_k_accuracy)*100
print(" Classification error for optimal K :"+ optimal_k_classification_error)


#Check for the formula of confidence interval
def confidenceinterval(classificationaccuracy,trainImagesSet ):
    kaccuracy = classificationaccuracy/100
    confidence_interval = []
    confidence_interval_pos = kaccuracy+1.96* (sqrt((kaccuracy*(1-kaccuracy))/len(trainImagesSet)))
    confidence_interval.append(confidence_interval_pos)
    confidence_interval_neg = kaccuracy-1.96* (sqrt((kaccuracy*(1-kaccuracy))/len(trainImagesSet)))
    confidence_interval.append(confidence_interval_neg)
    return confidence_interval


ci = confidenceinterval(optimal_k_classification_accuracy,trainImagesSet)
print("Confidence Interval for optimal k: ",ci)



#Confusion matrix calculation
def confusionmatrix(predictions, actuallabels):
    row = {}
    for i in range(len(actuallabels)):
        x = str(actuallabels[i]) + str(predictions[i])
        key = "group_{0}".format(x)
        if key in row:
            row["group_{0}".format(x)] = row["group_{0}".format(x)] + 1
        else:
            row["group_{0}".format(x)] = 1

    labelrows = []
    for x in range(0, 10):
        for y in range(0, 10):
            j = str(x) + str(y)
            p = "group_{0}".format(j)
            if p in row:
                labelrows.append(row["group_{0}".format(j)])
            else:
                labelrows.append(0)

    cm = reshape(labelrows, (10, 10))
    return cm

#Confusion matrix for optimal k predictions
cm = confusionmatrix(optimal_k_predictions,testLabelsSet)
print("confusion matrix for optimal-K : ",cm)


#Sliding Window code
sliding_win_predictor = MNISTPredictor(copyOfTrainingDataSet, optimal_k)
slinding_win_predictions = sliding_win_predictor(sliding_win_predictor, testImagesSet)


#Accuracy and error calculations
sliding_win_accuracy = evaluate_prediction(slinding_win_predictions, testLabelsSet)
sliding_win_classification_accuracy  = sliding_win_accuracy*100
print(" Sliding window classification accuracy for optimal K :"+ sliding_win_classification_accuracy)
sliding_win_classification_error = (1-sliding_win_accuracy)*100
print(" Sliding Window classification error for optimal K :"+ sliding_win_classification_error)

#confidence interval for sliding window
sliding_win_ci = confidenceinterval(sliding_win_classification_accuracy,trainImagesSet)
print("Sliding Window confidence Interval for optimal k: ",sliding_win_ci)

#Confusion matrix for sliding window predictions
sliding_win_cm = confusionmatrix(slinding_win_predictions,testLabelsSet)
print("confusion matrix for sliding window : ",sliding_win_cm)