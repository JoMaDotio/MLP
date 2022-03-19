import numpy as np
import matplotlib.pyplot as plt
from MLP import *

def MLP_binary_classification_2d(X, Y, net):
    plt.figure()
    for i in range(X.shape[1]):
        if Y[0,i] == 0:
            plt.plot(X[0,i], X[1, i], '.r')
        else:
            plt.plot(X[0,i], X[1, i], '.g')
    xmin, ymin = np.min(X[0,:]) - 0.5, np.min(X[1,:]) - 0.5
    xmax, ymax = np.max(X[0,:]) + 0.5, np.max(X[1,:]) + 0.5
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
    data = [xx.ravel(), yy.ravel()]
    zz = net.predict(data)
    zz = zz.reshape(xx.shape)
    plt.contourf(xx, yy, zz, alpha=0.8, cmap=plt.cm.RdBu)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.grid()
    plt.show()



if __name__ == "__main__":
    X = np.array([[0,0,1,1],[0,1,0,1]])
    Y = np.array([[1,0,0,1]])
    net = MLP((2,20,1))
    print(net.predict(X))
    MLP_binary_classification_2d(X, Y, net)
    net.train(X, Y)
    print(net.predict(X))
    MLP_binary_classification_2d(X, Y, net)