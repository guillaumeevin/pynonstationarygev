import numpy as np

from extreme_fit.distribution.abstract_params import AbstractParams


class AbstractExtremeParams(AbstractParams):
    SMALL_SCALE_PARAMETERS_ARE_UNDEFINED = True

    def __init__(self, loc: float, scale: float, shape: float):
        self.location = loc
        self.scale = scale
        self.shape = shape
        # By default, scale cannot be negative
        # (sometimes it happens, when we want to find a quantile for every point of a 2D map
        # then it can happen that a corner point that was not used for fitting correspond to a negative scale,
        # in the case we set all the parameters as equal to np.nan, and we will not display those points)
        self.has_undefined_parameters = (self.scale <= 0)
        if self.SMALL_SCALE_PARAMETERS_ARE_UNDEFINED:
            self.has_undefined_parameters = self.has_undefined_parameters or (np.isclose(self.scale, 0))