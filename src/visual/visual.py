import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from numpy import exp
from matplotlib.ticker import LinearLocator


def terrainMesh(peaks : np, posbound : np):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.arange(int(posbound[0][0]), int(posbound[1][0]), 1)
    Y = np.arange(int(posbound[0][1]), int(posbound[1][1]), 1)
    R = np.zeros((int(posbound[1][0]), int(posbound[1][1])))
    X, Y = np.meshgrid(X, Y)
    size = peaks.shape[0]
    for i in range(size):
        xx, yy, h, xi, yi = peaks[i]
        R += h * exp(-((X - xx) / xi) ** 2 - ((Y - yy) / yi) ** 2)
    surf = ax.plot_surface(X, Y, R, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(int(posbound[0][2]), int(posbound[1][2]))
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter('{x:.02f}')
    # plt.show()
    return fig, ax
    
def drawPath(path : np, peaks : np, posbound : np):
    fig, ax = terrainMesh(peaks, posbound)
    pos = np.rot90(path, -1)
    ax.plot(pos[0], pos[1], pos[2], color='r', zorder=99)
    plt.show()