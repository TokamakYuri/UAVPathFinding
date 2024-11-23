import numpy as np
from numpy import (
    inf
)

import utility.utility as ut

class individual:
    def __init__(self, pathnum) -> None:
        self.pos = np.zeros((pathnum, pathnum, pathnum))
        self.fitness = inf
        

        