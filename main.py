import numpy as np
import matplotlib.pyplot as plt
import pandas as pnd
from MLP import *


def MLP_binary_classification_2d(X, Y, net, title):
    plt.figure()
    for i in range(X.shape[1]):
        if Y[0, i] == 0:
            plt.plot(X[0, i], X[1, i], '.r')
        else:
            plt.plot(X[0, i], X[1, i], '.b')
    xmin, ymin = np.min(X[0, :]) - 0.5, np.min(X[1, :]) - 0.5
    xmax, ymax = np.max(X[0, :]) + 0.5, np.max(X[1, :]) + 0.5
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100),
                         np.linspace(ymin, ymax, 100))
    data = [xx.ravel(), yy.ravel()]
    zz = net.predict(data)
    zz = zz.reshape(xx.shape)
    plt.contourf(xx, yy, zz, alpha=0.8, cmap=plt.cm.RdBu)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.grid()
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    #file = pnd.read_csv("XOR.csv")
    files = ["XOR.csv", "moons.csv", "circles.csv", "blobs.csv"]
    titles = ["XOR", "Moons", "Circles", "Blobs"]

    file = pnd.read_csv(files[3])
    x1 = np.array(file.x1).reshape(1, -1)
    x2 = np.array(file.x2).reshape(1, -1)
    X = np.concatenate([x1, x2])
    Y = np.array(file.y).reshape(1, -1)
    net = MLP((2, 3, 1))
    print(net.predict(X))
    title1 = f"MLP pre-training: {titles[3]} problem"
    title2 = f"MLP post-training: {titles[3]} problem"
    MLP_binary_classification_2d(X, Y, net, title1)
    net.train(X, Y)
    print(net.predict(X))
    MLP_binary_classification_2d(X, Y, net, title2)
