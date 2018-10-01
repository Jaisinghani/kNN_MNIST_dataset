# from numpy import *
# import matplotlib.pyplot as plt
# #Confusion matrix calculation
# def confusionmatrix(predictions,actuallabels):
#     row ={}
#     for i in range(len(actuallabels)):
#         x = str(actuallabels[i]) + str(predictions[i])
#         key = "group_{0}".format(x)
#         if key in row:
#             row["group_{0}".format(x)] = row["group_{0}".format(x)] + 1
#         else:
#             row["group_{0}".format(x)] = 1
#
#     labelrows = []
#     for x in range(0,10):
#         for y in range(0,10):
#             j = str(x)+str(y)
#             p = "group_{0}".format(j)
#             if p in row:
#                 labelrows.append(row["group_{0}".format(j)])
#             else:
#                 labelrows.append(0)
#
#     cm = reshape(labelrows,(10,10))
#     return cm
#
# actuallabels = [0,1,2,3,4,5,6,7,8,9,9,6,5,2,4,0,3,7,9,8,1,1,0,3,5,4]
# predictions =  [0,9,0,8,6,5,3,2,8,9,9,6,5,0,1,0,2,7,8,9,1,2,3,4,5,4]
# con = confusionmatrix(predictions,actuallabels)
# print ("matrix :",con)
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(con)
# plt.title('Confusion matrix of the classifier')
# fig.colorbar(cax)
# ax.set_xticklabels([''] + actuallabels)
# ax.set_yticklabels([''] + actuallabels)
# plt.xlabel('Predicted')
# plt.ylabel('actuallabels')
# plt.show()


# import struct
# from numpy import *
# import numpy as np
# import matplotlib.pyplot as plt
# def parse_labels(fileName):
#     """Parse labels from the binary file."""
#     filePath = "./dataset/" + fileName
#     with open(filePath, "rb") as binary_file:
#         data = binary_file.read()
#     magic, n = struct.unpack_from('>2i', data)
#     assert magic == 2049, "Wrong magic number: %d" % magic
#
#     labels = struct.unpack_from('>%dB' % n, data, offset=8)
#     return labels
#
#
# def parse_images(fileName):
#     """Parse images from the binary file."""
#     filePath= "./dataset/"+fileName
#     with open(filePath, "rb") as binary_file:
#         data = binary_file.read()
#
#     magic, n, rows, cols = struct.unpack_from('>4i', data)
#     assert magic == 2051, "Wrong magic number: %d" % magic
#
#     num_pixels = n * rows * cols
#     pixels = struct.unpack_from('>%dB' % num_pixels, data, offset=16)
#
#
#     pixels = asarray(pixels, dtype=np.float)
#
#
#     images = pixels.reshape((n, cols, rows))
#     return images
#
#
# print("loading training images")
# trainImagesSet = parse_images("train-images-idx3-ubyte")
#
# print("extra space              ")
#
# # a = np.array([[ 1.,  1.,  1.,  1.,  1., 2., 3.],
# #                [ 1.,  1.,  1.,  1.,  1., 4., 5.],
# #                [ 1.,  1.,  1.,  1.,  1., 7., 8.]])
# print("t  :", trainImagesSet[0].shape)
# padded = np.pad(trainImagesSet[0], [(1, 1),(1, 1)], mode='constant')
# print("padded  :", padded.shape)
#
#
#
#
# for i in range(3):
#     for j in range(3):
#         paddedImage = padded[i:i+28, j:j+28]
#         #print("padded sliced image :",paddedImage )
#         print("padded sliced image :", paddedImage.shape)


# import numpy as np
# a = np.array([[ 1.,  2.],
#              [3.,  4.]])
# print("t  :", a)
# print("t  :", a.shape)
# padded = np.pad(a, [(1, 1),(1, 1)], mode='constant')
# print("padded  :", padded)
# print("padded  :", padded.shape)
#
# for i in range(3):
#     for j in range(3):
#         slicedImage = padded[i:i + 2, j:j + 2]
#         print("slicedImage :", slicedImage)
#         print("slicedImage :", slicedImage.shape)

# import datetime
# print(datetime.datetime.now())
# x = [10,11,12,13,14,15,16,17,18,19,20]
# for i in range(5,10):
#     print("i :",i)
#     print("x :",x[i])
# k_accuracy = [(1, 97.26333333333332), (2, 97.28333333333333), (3, 97.41000000000002), (4, 97.45333333333333), (5, 97.29166666666668), (6, 97.17666666666667), (7, 97.12166666666666), (8, 97.09033333333335), (9, 97.00833333333334), (10, 96.91500000000002)]
# print("Accuracies for all k ranging from 1 to 10 is :", k_accuracy)
# k_accuracy.sort(key=lambda val: val[1])
# print("k_accuracy :", k_accuracy[-1][0])

# import numpy as np
# def confidenceinterval(classificationaccuracy,trainImagesSet ):
#     kaccuracy = classificationaccuracy/100
#     confidence_interval = []
#     confidence_interval_pos = kaccuracy+1.96* (np.sqrt((kaccuracy*(1-kaccuracy))/trainImagesSet))
#     confidence_interval.append(confidence_interval_pos)
#     confidence_interval_neg = kaccuracy-1.96* (np.sqrt((kaccuracy*(1-kaccuracy))/trainImagesSet))
#     confidence_interval.append(confidence_interval_neg)
#     return confidence_interval
#
# ci = confidenceinterval(97.16, 10000)
# print("ci :", ci)
import numpy as np
ok = 0.9716
slw = 0.9853
total = 20000
diff = slw - ok
print("difference :", diff)
mean = (ok+slw)/total
print("mean :", mean)
sd = 1.96 * (np.sqrt((mean *(1-mean))/total))
print("h :", sd)