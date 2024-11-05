from numpy import pi as PI

import numpy as np
from numpy import (
    dot, abs
)
from numpy.linalg import norm as dis

from math import (
    sin, cos, acos
)

def rotate90MatUP(array : np, times : int = 1):
    r, c = array.shape
    zs = np.zeros((max(array.shape) - min(array.shape), max(array.shape)))
    tmp = np.vstack([array, zs])
    tmp = np.rot90(tmp, times)
    return tmp[:c, :r]

def getPathLength(path : np) -> float:
    size = path.shape[0]
    length = 0.
    for i in range(size - 1):
        length += dis(path[i + 1] - path[i])
    return length

def getPathAngel(path : np) -> float:
    size = path.shape[0]
    angel = 0.
    for i in range(size - 2):
        vec_1 = path[i + 1] - path[i]
        vec_2 = path[i + 2] - path[i + 1]
        angel += acos(dot(vec_1, vec_2)/abs(dis(vec_1) * dis(vec_2))) / PI * 180.
    return angel

def getPathRadar(path : np) -> float:
    return 0.

def calaTerrainHeight(pos : np, settings : dict) -> float:
    return 0.

def checkCollision(pos : np, settings : dict, terrain : np = None) -> bool:
    x, y, z = pos[0], pos[1], pos[2]
    if terrain != None:
        if terrain[x, y] > z:
            return True
        else:
            return False
    else:
        height = calaTerrainHeight(pos, settings)
        if height > z:
            return True
        else:
            return False
        
def calaSphericPosition(start: np, svector : np) -> np:
    r, s, p = svector[0], svector[1], svector[2]
    vec = np.array([r * sin(s) * cos(p), r * sin(s) * sin(p), r * cos(s)])
    pos = np.add(start, vec)
    return pos

def calaCost(path : np, weight : dict) -> float:
    length = getPathLength(path)
    angel = getPathAngel(path)
    radar = getPathRadar(path)
    cost = weight.get('l') * length + weight.get('a') * angel + weight.get('r') * radar
    return cost

def calaFitness(path : np, weight: dict, settings : np, terrain : np = None) -> float:
    cost = calaCost(path, weight)
    for i in range(path.shape[0]):
        if checkCollision(path[i], settings, terrain):
            return cost * 1000.
    return cost