from extreme_fit.distribution.abstract_params import AbstractParams
from extreme_fit.model.utils import r


class GpdParams(AbstractParams):
    # TODO: understand better why the gpdfit return 2 parameters, alors que d'un autre cote d autres definitions de la distribution parlent d un parametre location

    # Parameters
    PARAM_NAMES = [AbstractParams.SCALE, AbstractParams.SHAPE]
    # Summary
    SUMMARY_NAMES = PARAM_NAMES + AbstractParams.QUANTILE_NAMES

    def __init__(self, scale: float, shape: float, threshold: float = None):
        super().__init__(loc=0.0, scale=scale, shape=shape)
        self.threshold = threshold

    def quantile(self, p) -> float:
        return r.qgpd(p, self.location, self.scale, self.shape)[0] + self.threshold

    @property
    def param_values(self):
        return [self.scale, self.shape]
