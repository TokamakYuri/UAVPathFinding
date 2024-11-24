import numpy as np
import numpy.random as nprng
import matplotlib.pyplot as plt

from utility import utility as ut
from pso import pso
from visual import visual as vs

rng = nprng.default_rng()

posbound = np.array([[0., 0., 0.],
                     [100., 100., 100.]])
backwardveltimes = 0.5
forwardveltimes = 0.5
velbound = np.array([-backwardveltimes * (posbound[1] - posbound[0]), 
                     forwardveltimes * (posbound[1] - posbound[0])])
weibound = np.array([0.4, 0.9])
peaks = np.array([[20.,20.,45.,7.,9.],
                  [30.,60.,51.,5.,5.],
                  [50.,50.,62.,9.,6.],
                  [70.,20.,88.,3.,4.],
                  [85.,85.,40.,4.,7.],
                  [60.,70.,74.,4.,5.]])
radar = np.array([[80., 30., 0.],
                  [30., 80., 0.]])
radarsettings = {'max' : 30,
                 'min' : 20,
                 'vs' : 20}
start = np.array([0., 0., 10.])
stop = np.array([100., 100., 60.])

psooption = {'num' : 20,
             'step' : 2000,
             'pathnum' : 20,
             'weight' : {'l' : 0.5, 'r' : 0.3, 'a': 0.2},
             'c1' : 1.5,
             'c2' : 1.5,
             'w' : 0.8}

psobest = pso.pso(start, stop, posbound, velbound, psooption, peaks, radar, radarsettings, rng=rng)
fig, ax = vs.draw(psobest.bestpath, peaks, posbound, radar, radarsettings)
# ldpsobest = pso.ldpso(start, stop, posbound, velbound, weibound, psooption, peaks, rng=rng)
# fig, ax = vs.draw(ldpsobest.bestpath, peaks, posbound, radar, radarsettings)
plt.show()