import numpy  as np

import utility.utility as ut

class individual:
    def __init__(self, ranges) -> None:
        
        self.range = np.array(ranges)
        self.dimension = self.range.shape[0]

        