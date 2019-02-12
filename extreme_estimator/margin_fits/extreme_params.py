from extreme_estimator.margin_fits.abstract_params import AbstractParams


class ExtremeParams(AbstractParams):

    # Extreme parameters
    SCALE = 'scale'
    LOC = 'loc'
    SHAPE = 'shape'
    PARAM_NAMES = [LOC, SCALE, SHAPE]

    def __init__(self, loc: float, scale: float, shape: float):
        self.location = loc
        self.scale = scale
        self.shape = shape
        assert self.scale > 0

    @property
    def param_values(self):
        return [self.location, self.scale, self.shape]