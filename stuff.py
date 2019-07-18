import numpy as np
from scipy.stats import norm
from sklearn.pipeline import Pipeline
import numpy as np

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

class Pipeline_2(Pipeline):
    def just_transforms(self, X):
        """
        Applies all transforms to the data, without applying last estimator.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step of the pipeline.
            """

        Xt = X
        for name, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)
        return Xt

def plot_decision(X, y, fitted_pipe, cmap = 'coolwarm', y_label='$P(y=1)$', lab_x1 = '$X_1$', lab_x2 = '$X_2$'):
    '''
    Plot the decision function of a scikit-learn estimator wrapped in a pipeline object 
    implementing just_transforms method (like Pipeline_2). X has to be 2 dimensional OR include a dimension reduction step.
    '''
    normalized = pipe.just_transforms(all_X)
    cmap = cmap
    lims_low = normalized.min(axis = 0) + (-.01, .01)
    lims_high = normalized.max(axis = 0)+ (-.01, .01)
    xx, yy = np.mgrid[lims_low[0]:lims_high[0]+.01:.01, lims_low[1]:lims_high[1]+.01:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = pipe.steps[-1][1].predict_proba(grid)[:, 1].reshape(xx.shape)

    f, ax = plt.subplots(figsize=(15, 10))
    contour = ax.contourf(xx, yy, probs, 25, cmap=cmap,
                                  vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label(y_label)
    ax_c.set_ticks([0, .25, .5, .75, 1])

    ax.scatter(normalized[:, 0], normalized[:, 1], c=y, s=50,
                       cmap=cmap, edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
                   xlim=(lims_low[0], lims_high[0]), ylim=(lims_low[1], lims_high[1]),
                          xlabel=lab_x1, ylabel=lab_x2 
    return f
