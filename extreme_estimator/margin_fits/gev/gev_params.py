from collections import OrderedDict

from cached_property import cached_property

from extreme_estimator.extreme_models.utils import r
from extreme_estimator.margin_fits.extreme_params import ExtremeParams
import numpy as np
from scipy.special import gamma


class GevParams(ExtremeParams):
    # Parameters
    PARAM_NAMES = [ExtremeParams.LOC, ExtremeParams.SCALE, ExtremeParams.SHAPE]
    # Summary
    SUMMARY_NAMES = PARAM_NAMES + ExtremeParams.QUANTILE_NAMES
    NB_SUMMARY_NAMES = len(SUMMARY_NAMES)

    def __init__(self, loc: float, scale: float, shape: float, block_size: int = None):
        super().__init__(loc, scale, shape)
        self.block_size = block_size

    def quantile(self, p) -> float:
        if self.has_undefined_parameters:
            return np.nan
        else:
            return r.qgev(p, self.location, self.scale, self.shape)[0]

    def density(self, x):
        if self.has_undefined_parameters:
            return np.nan
        else:
            res = r.dgev(x, self.location, self.scale, self.shape)
            if isinstance(x, float):
                return res[0]
            else:
                return np.array(res)

    @property
    def param_values(self):
        if self.has_undefined_parameters:
            return [np.nan for _ in range(3)]
        else:
            return [self.location, self.scale, self.shape]

    def __str__(self):
        return self.to_dict().__str__()

    def quantile_strength_evolution_ratio(self, p=0.99, mu1=0.0, sigma1=0.0):
        """
        Compute the relative evolution of some quantile with respect to time.
        (when mu1 and sigma1 can be modified with time)

        :param p: level of the quantile
        :param mu1: temporal slope of the location parameter
        :param sigma1: temporal slope of the scale parameter
        :return: A string summarizing the evolution ratio
        """
        initial_quantile = self.quantile(p)
        quantity_increased = mu1
        if sigma1 != 0:
            quantity_increased += (sigma1 / self.shape) * (1 - (- np.float_power(np.log(p), -self.shape)))
        return quantity_increased / initial_quantile

    # Compute some indicators (such as the mean and the variance)

    def g(self, k):
        # Compute the g_k parameters as defined in wikipedia
        # https://fr.wikipedia.org/wiki/Loi_d%27extremum_g%C3%A9n%C3%A9ralis%C3%A9e
        return gamma(1 - k * self.shape)

    @property
    def mean(self):
        if self.shape >= 1:
            return np.inf
        else:
            return self.location + self.scale * (self.g(k=1) - 1) / self.shape

    @property
    def variance(self):
        if self.shape >= 0.5:
            return np.inf
        else:
            return ((self.scale / self.shape) ** 2) * (self.g(k=2) - self.g(k=1) ** 2)

    @property
    def std(self):
        return np.sqrt(self.variance)

    @classmethod
    def indicator_names(cls):
        return ['mean', 'std'] + cls.QUANTILE_NAMES[:2]

    @cached_property
    def indicator_name_to_value(self) -> OrderedDict:
        indicator_name_to_value = OrderedDict()
        indicator_name_to_value['mean'] = self.mean
        indicator_name_to_value['std'] = self.std
        for quantile_name, quantile_value in zip(self.QUANTILE_NAMES[:2], self.QUANTILE_P_VALUES):
            indicator_name_to_value[quantile_name] = self.quantile(quantile_value)
        assert all([a == b for a, b in zip(self.indicator_names(), indicator_name_to_value.keys())])
        return indicator_name_to_value
