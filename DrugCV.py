import cv2
import os
import numpy as np
import time

# READ IN ALL THE SHAPE FILES INTO A LIST

file_names = []
test_file_names = []
filename = "C:/Users/welln/OneDrive/Research/Li_Lab/OPENCV/conform/"
test_file = None

for f in os.listdir(filename):
    if f.endswith(".png"):
        file_id = f.split("_")[0]
        file_names.append((file_id, f))

# file_names.sort(key=lambda x: int(x[0]))

if test_file is not None:
    for f in os.listdir(test_file):
        if f.endswith(".png"):
            file_id = f.split("_")[0]
            test_file_names.append((file_id, f))

    test_file_names.sort(key=lambda x: int(x[0]))

# READ IN ALL THE IMAGES AND MAKE THEM BINARY

images = []
for f in file_names:
    im = cv2.imread(filename + f[1], cv2.IMREAD_GRAYSCALE)
    images.append((f[0], im))

if test_file is not None:
    test_images = []
    for f in test_file_names:
        im = cv2.imread(test_file + f[1], cv2.IMREAD_GRAYSCALE)
        _, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
        test_images.append((f[0], im))

# COMPARE THE DISTANCE OF EACH IMAGE WITH EACH OTHER


compare = np.zeros(shape=(len(file_names), len(file_names)))

for x in range(len(file_names)):
    for y in range(len(file_names)):
        ca, _ = cv2.findContours(images[x][1], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cb, _ = cv2.findContours(images[y][1], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        compare[x][y] = cv2.matchShapes(ca, cb, cv2.CONTOURS_MATCH_I3, 0)

# COMPARE EACH OF THE TEST IMAGES TO THE OTHERS AND FIND THE MOST SIMILAR
if test_file is not None:
    results = [["TestId", "BestMatchID", "Distance"]]
    for img in test_images:
        curr_results = []
        for img2 in images:
            result = cv2.matchShapes(img[1], img2[1], cv2.CONTOURS_MATCH_I2, 0)
            curr_results.append([img[0], img2[0], result])
        best_fit = min(curr_results, key=lambda x: x[2])
        results.append(best_fit)

np.savetxt('results4_noscale.csv', compare, delimiter=',')

distance_context = np.zeros(shape=(len(file_names), len(file_names)))
distance_Haus = np.zeros(shape=(len(file_names), len(file_names)))

"""
for x in range(len(file_names)):
    for y in range(x):
        ca, _ = cv2.findContours(images[x][1], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cb, _ = cv2.findContours(images[y][1], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        hd = cv2.createHausdorffDistanceExtractor()
        sd = cv2.createShapeContextDistanceExtractor()

        d1 = hd.computeDistance(ca[0], cb[0])
        # d2 = sd.computeDistance(ca[0], cb[0])

        # distance_context[x][y] = d2
        distance_Haus[x][y] = d1
        print(x, y)

np.savetxt('results_haus_noscale_complete.csv', distance_Haus, delimiter=',')
# np.savetxt('results_context_noscale_complete.csv', distance_context, delimiter=',')
"""

for i in range(compare.shape[0]):
    mins = compare[i].argsort()[1:4]
    labels = [1 if x > 21 else 0 for x in mins]
    zeros = np.bincount(labels)[0]
    ones = np.bincount(labels)[1]
    print(zeros, ones)
    if i > 21:
        real = "1"
    else:
        real = "0"
    if zeros > ones:
        pred = "0"
    else:
        pred = "1"
    print("Real: {a}   Pred: {b}".format(a=real, b=pred))
