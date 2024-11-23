from numpy import pi as PI

import numpy as np
from numpy import (
    dot, abs, inf
)
from numpy.linalg import norm as dis

from math import (
    sin, cos, acos, exp
)

def getPathLength(path : np) -> float:
    size = path.shape[0]
    length = 0.
    for i in range(size - 1):
        length += dis(path[i + 1] - path[i])
    return length

def getPathAngle(path : np) -> float:
    size = path.shape[0]
    angle = 0.
    for i in range(size - 2):
        vec_1 = path[i + 1] - path[i]
        vec_2 = path[i + 2] - path[i + 1]
        try:
            angle += acos(dot(vec_1, vec_2)/abs(dis(vec_1) * dis(vec_2)))
        except:
            if abs(dis(vec_1) * dis(vec_2)) == 0:
                angle += 0
    return angle

def getPathRadar(path : np, radar : np, radarsettings : dict) -> float:
    size = radar.shape[0]
    pathnum = path.shape[0]
    score = 0.
    radarmax = radarsettings.get('max')
    radarmin = radarsettings.get('min')
    for i in range(size):
        for j in range(pathnum):
            distance = dis(path[j] - radar[i])
            if distance <= radarmin:
                score += 100.
            elif distance <= 30:
                score += 100. / distance
    return score

def calaTerrainHeight(pos : np, settings : np) -> float:
    x, y, z = pos
    height = 0.
    for i in range(settings.shape[0]):
        xx, yy, h, xi, yi = settings[i]
        height += h * exp(-((x - xx) / xi) ** 2 - ((y - yy) / yi) ** 2)
    return height

def checkCollision(pos : np, settings : np, terrain : np = None) -> bool:
    x, y, z = pos
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
    r, s, p = svector
    vec = np.array([r * sin(s) * cos(p), r * sin(s) * sin(p), r * cos(s)])
    pos = np.add(start, vec)
    return pos

def calaCost(path : np, weight : dict, radar : np, radarsettings : dict) -> float:
    length = getPathLength(path)
    angle = getPathAngle(path)
    radar = getPathRadar(path, radar, radarsettings)
    cost = weight.get('l') * length + weight.get('a') * angle + weight.get('r') * radar
    return cost

def calaFitness(path : np, weight: dict, settings : np, radar : np, radarsettings : dict, terrain : np = None) -> float:
    cost = calaCost(path, weight, radar, radarsettings)
    for i in range(path.shape[0]):
        if checkCollision(path[i], settings, terrain):
            return cost * 1000.
    return cost

def calaLengthArray(path : np) -> np:
    size = path.shape[0]
    length_array = np.zeros((size, size))
    for i in range(size):
        for j in range(size - i):
            distance = dis(path[i] - path[j])
            length_array[i][j] = distance
            length_array[j][i] = distance
    return length_array

def hamiltonPath(graph : np):
    size = graph.shape[0]
    length = inf
    