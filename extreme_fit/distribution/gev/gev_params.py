from collections import OrderedDict
from typing import List

from cached_property import cached_property
from mpmath import euler, pi

from extreme_fit.distribution.abstract_params import AbstractParams
from extreme_fit.model.utils import r
import numpy as np
from scipy.special import gamma


class GevParams(AbstractParams):
    # Parameters
    PARAM_NAMES = [AbstractParams.LOC, AbstractParams.SCALE, AbstractParams.SHAPE]
    # Summary
    SUMMARY_NAMES = PARAM_NAMES + AbstractParams.QUANTILE_NAMES
    NB_SUMMARY_NAMES = len(SUMMARY_NAMES)

    def __init__(self, loc: float, scale: float, shape: float, block_size: int = None, accept_zero_scale_parameter=False):
        super().__init__(loc, scale, shape)
        self.block_size = block_size
        if accept_zero_scale_parameter and scale == 0.0:
            self.has_undefined_parameters = False

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

    def quantile_strength_evolution(self, p=0.99, mu1=0.0, sigma1=0.0):
        # todo: chagne the name to time derivative
        """
        Compute the variation of some quantile with respect to time.
        (when mu1 and sigma1 can be modified with time)

        :param p: level of the quantile
        :param mu1: temporal slope of the location parameter
        :param sigma1: temporal slope of the scale parameter
        :return: A float that equals evolution ratio
        """
        quantile_annual_variation = mu1
        if sigma1 != 0:
            power = np.float_power(- np.log(p), -self.shape)
            quantile_annual_variation -= (sigma1 / self.shape) * (1 - power)
        return quantile_annual_variation

    # Compute some indicators (such as the mean and the variance)

    def g(self, k) -> float:
        # Compute the g_k parameters as defined in wikipedia
        # https://fr.wikipedia.org/wiki/Loi_d%27extremum_g%C3%A9n%C3%A9ralis%C3%A9e
        return gamma(1 - k * self.shape)

    @property
    def mean(self) -> float:
        if self.has_undefined_parameters:
            return np.nan
        elif self.shape >= 1:
            return np.inf
        elif self.shape == 0:
            return self.location + self.scale * euler
        else:
            return self.location + self.scale * (self.g(k=1) - 1) / self.shape

    @property
    def variance(self) -> float:
        if self.has_undefined_parameters:
            return np.nan
        elif self.shape >= 0.5:
            return np.inf
        elif self.shape == 0.0:
            return (self.scale * pi) ** 2 / 6
        else:
            return ((self.scale / self.shape) ** 2) * (self.g(k=2) - self.g(k=1) ** 2)

    @property
    def std(self) -> float:
        return np.sqrt(self.variance)

    @classmethod
    def indicator_names(cls) -> List[str]:
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

    @classmethod
    def greek_letter_from_gev_param_name(cls, gev_param_name):
        assert gev_param_name in cls.PARAM_NAMES
        return {
            cls.LOC: 'mu',
            cls.SCALE: 'sigma',
            cls.SHAPE: 'zeta',
        }[gev_param_name]