#!/usr/bin/env python3.6

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from svm import HarringtonSmoClassifier as HSC
from sklearn.metrics import accuracy_score


def plot(X, Y):
    data = np.concatenate((np.matrix(X), np.matrix(Y).T), axis=1)
    df = pd.DataFrame(data=data, columns=['X0', 'X1', 'Y'])
    sns.lmplot('X0', 'X1', data=df, hue='Y', fit_reg=False)
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

    
df = pd.read_csv('linearly_separable.csv', header=None)
df = np.random.permutation(df)
X = df[:,:2]
Y = df[:,2]

total_acc = 0
iter = 30

clf = HSC()

for i in range(iter):

    split = int(len(Y) * 0.8)
    train_x = X[:split,:]
    train_y = Y[:split]
    test_x = X[split:,:]
    test_y = Y[split:]
    #train_x = X
    #train_y = Y
    #test_x = X
    #test_y = Y

    clf.fit(train_x, train_y)
    print('weights', clf.w)

    hypotheses = clf.predict(test_x)
    print('hypotheses:', hypotheses)

    acc = accuracy_score(test_y, hypotheses)
    print('Accuracy:', acc)
    total_acc += acc

    print_confusion_matrix(test_y, hypotheses)

print("Average accuracy: %.3f" % (total_acc/iter))

#plot(X, Y)
