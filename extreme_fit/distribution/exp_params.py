from extreme_fit.distribution.abstract_params import AbstractParams


class ExpParams(object):
    PARAM_NAMES = [AbstractParams.SCALE]

    def __init__(self, scale: float):
        self.scale = scale

