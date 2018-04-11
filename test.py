#!/usr/bin/env python3.6

import matplotlib.pyplot as plt
import numpy as np
from svm import HarringtonSmoClassifier as HSC


def get_data(filename):
    datamatrix = []
    labelmatrix = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        datamatrix.append([float(lineArr[0]), float(lineArr[1])])
        labelmatrix.append(int(lineArr[2]))
    return datamatrix, labelmatrix


def plot_fit(fit_line, datamatrix, labelmatrix):
    weights = fit_line.getA()

    dataarray = np.asarray(datamatrix)
    n = dataarray.shape[0]

    # Keep track of the two classes in different arrays so they can be plotted later...
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelmatrix[i]) == 1:
            xcord1.append(dataarray[i, 1])
            ycord1.append(dataarray[i, 2])
        else:
            xcord2.append(dataarray[i, 1])
            ycord2.append(dataarray[i, 2])
    fig = plt.figure()

    # Plot the data as points with different colours
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    # Plot the best-fit line
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def accuracy(labels, hypotheses):
    count = 0.0
    correct = 0.0

    for l, h in zip(labels, hypotheses):
        count += 1.0
        if l == h:
            correct += 1.0
    return correct / count


def print_confusion_matrix(labels, hypotheses):
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    count = 1.0
    for l, h in zip(labels, hypotheses):
        count += 1.0
        if l == 1 and h == 1:
            tp += 1.0
        elif l == 1 and h == -1:
            fn += 1.0
        elif l == -1 and h == -1:
            tn += 1.0
        else:
            fn += 1
    print('-----------------------------')
    print('\tConfusion Matrix')
    print('-----------------------------')
    print('\t\tPredicted')
    print('\tActual\t-1\t+1')
    print('-----------------------------')
    print('\t-1\t', tn, '\t', fp)
    print('-----------------------------')
    print('\t+1\t', fn, '\t', tp)
    print('-----------------------------')

    
X, Y = get_data('linearly_separable.tsv')
split = int(len(Y) * 0.8)
train_x = X[:split]
train_y = Y[:split]
test_x = X[split:]
test_y = Y[split:]

clf = HSC()
b, alphas, sv = clf.fit(train_x, train_y)
print('b:', b)
print('alphas:', alphas)
print('support vectors:', sv)
#plot_fit(w, X, Y)

hypotheses = clf.predict(test_x)

print('Accuracy:', accuracy(test_y, hypotheses))

print_confusion_matrix(test_y, hypotheses)

