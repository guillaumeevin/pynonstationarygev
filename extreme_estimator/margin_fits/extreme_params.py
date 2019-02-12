from abc import ABC

from extreme_estimator.margin_fits.abstract_params import AbstractParams


class ExtremeParams(AbstractParams, ABC):
    # Extreme parameters
    SCALE = 'scale'
    LOC = 'loc'
    SHAPE = 'shape'

    def __init__(self, loc: float, scale: float, shape: float):
        self.location = loc
        self.scale = scale
        self.shape = shape
        assert self.scale > 0
