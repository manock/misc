import numpy as np
from scipy.stats import norm

class ecdf():
    # R-like ecdf class
    def __init__(self, x, winsorized = True):
        self.base = np.sort(x)
        self.winsorized = winsorized
        self.n = x.shape[0]
        self.lims = 1 / (4*self.n**(1/4) * np.sqrt(np.pi * np.log(self.n)))
    def __repr__(self):
        return str('EDF constructed on object with dims ') + str(self.base.shape)
    def __call__(self, y):
        self.probs = np.searchsorted(self.base, y) / self.n
        if self.winsorized:
            self.probs[(self.probs<self.lims)] = self.lims
            self.probs[self.probs>1-self.lims] = 1 - self.lims
        self.normal_mappings = norm.ppf(self.probs)
        return self