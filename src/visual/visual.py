import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from numpy import (
    exp, sin, cos
)

from numpy import (
    pi
)

def terrainMesh(peaks : np, posbound : np, ax):
    X = np.arange(int(posbound[0][0]), int(posbound[1][0]), 1)
    Y = np.arange(int(posbound[0][1]), int(posbound[1][1]), 1)
    Z = np.zeros((int(posbound[1][0]), int(posbound[1][1])))
    X, Y = np.meshgrid(X, Y)
    size = peaks.shape[0]
    for i in range(size):
        xx, yy, h, xi, yi = peaks[i]
        Z += h * exp(-((X - xx) / xi) ** 2 - ((Y - yy) / yi) ** 2)
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, zorder=0)
    
def drawPath(path : np, ax):
    pos = np.rot90(path, -1)
    ax.plot(pos[0], pos[1], pos[2], color='r', zorder=99)

def drawRadar(radar : np, radarsettings : dict, ax):
    radarvs = radarsettings.get('vs')
    for i in range(radar.shape[0]):
        u = np.linspace(0, 2 * pi, 100)
        v = np.linspace(0, pi / 2., 100)
        x = radarvs * np.outer(cos(u), sin(v)) + radar[i][0]
        y = radarvs * np.outer(sin(u), sin(v)) + radar[i][1]
        z = radarvs * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, zorder=99)

def draw(path : np, peaks : np, posbound : np, radar : np, radarsettings : dict):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_zlim(int(posbound[0][2]), int(posbound[1][2]))
    terrainMesh(peaks, posbound, ax)
    drawPath(path, ax)
    # drawRadar(radar, radarsettings, ax)
    return fig, ax