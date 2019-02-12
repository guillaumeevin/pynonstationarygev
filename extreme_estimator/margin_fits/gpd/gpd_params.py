from extreme_estimator.extreme_models.utils import r
from extreme_estimator.margin_fits.extreme_params import ExtremeParams


class GpdParams(ExtremeParams):
    # TODO: understand better why the gpdfit return 2 parameters, alors que d'un autre cote d autres definitions de la distribution parlent d un parametre location

    # Parameters
    PARAM_NAMES = [ExtremeParams.SCALE, ExtremeParams.SHAPE]
    # Summary
    SUMMARY_NAMES = PARAM_NAMES + ExtremeParams.QUANTILE_NAMES

    def __init__(self, scale: float, shape: float, threshold: float = None):
        super().__init__(loc=0.0, scale=scale, shape=shape)
        self.threshold = threshold

    def quantile(self, p) -> float:
        return r.qgpd(p, self.location, self.scale, self.shape)[0] + self.threshold

    @property
    def param_values(self):
        return [self.scale, self.shape]
