import numpy as np
import numpy.random as nprng

from utility import utility as ut
from pso import pso
from visual import visual as vs

rng = nprng.default_rng()

posbound = np.array([[0., 0., 0.], [100., 100., 100.]])
backwardveltimes = 0.1
forwardveltimes = 0.1
velbound = np.array([-backwardveltimes * (posbound[1] - posbound[0]), 
                     forwardveltimes * (posbound[1] - posbound[0])])
weibound = np.array([0.4, 0.9])
peaks = np.array([[20.,20.,45.,7.,9.],
                  [30.,60.,51.,5.,5.],
                  [50.,50.,62.,9.,6.],
                  [70.,20.,88.,3.,4.],
                  [85.,85.,40.,4.,7.],
                  [60.,70.,74.,4.,5.]])
start = np.array([0., 0., 10.])
stop = np.array([100., 100., 60.])

psooption = {'num' : 20,
             'step' : 3000,
             'pathnum' : 20,
             'weight' : {'l' : 1.0, 'r' : 0.3, 'a': 0.0},
             'c1' : 1.5,
             'c2' : 1.5,
             'w' : 0.8}

# psobest = pso.pso(start, stop, posbound, velbound, psooption, peaks, rng=rng)
# print(psobest.bestpath)
# vs.drawPath(psobest.bestpath, peaks, posbound)
ldpsobest = pso.ldpso(start, stop, posbound, velbound, weibound, psooption, peaks, rng=rng)
print(ldpsobest.bestpath)
vs.drawPath(ldpsobest.bestpath, peaks, posbound)