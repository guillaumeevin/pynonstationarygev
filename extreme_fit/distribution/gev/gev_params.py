from collections import OrderedDict
import matplotlib.pyplot as plt
from typing import List

from cached_property import cached_property
from mpmath import euler

from extreme_fit.distribution.abstract_extreme_params import AbstractExtremeParams
from extreme_fit.distribution.abstract_params import AbstractParams
from extreme_fit.model.utils import r
import numpy as np
from scipy.special import gamma


class GevParams(AbstractExtremeParams):
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

    def sample(self, n) -> np.ndarray:
        if self.has_undefined_parameters:
            return np.nan
        else:
            return np.array(r.rgev(n, self.location, self.scale, self.shape))

    def quantile(self, p) -> float:
        if self.has_undefined_parameters:
            return np.nan
        else:
            return r.qgev(p, self.location, self.scale, self.shape)[0]

    def return_level(self, return_period):
        return self.quantile(1 - 1 / return_period)

    def density(self, x, log_scale=False):
        if self.has_undefined_parameters:
            return np.nan
        else:
            res = r.dgev(x, self.location, self.scale, self.shape, log_scale)
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

    def time_derivative_of_return_level(self, p=0.99, mu1=0.0, sigma1=0.0):
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
            if self.shape == 0:
                quantile_annual_variation -= sigma1 * np.log(- np.log(p))
            else:
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
            mean = np.nan
        elif self.shape >= 1:
            mean = np.inf
        elif self.shape == 0:
            mean = self.location + self.scale * float(euler)
        else:
            mean = self.location + self.scale * (self.g(k=1) - 1) / self.shape
        assert isinstance(mean, float)
        return mean

    @property
    def variance(self) -> float:
        if self.has_undefined_parameters:
            return np.nan
        elif self.shape >= 0.5:
            return np.inf
        elif self.shape == 0.0:
            return (self.scale * np.pi) ** 2 / 6
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
    def greek_letter_from_param_name(cls, param_name):
        assert param_name in cls.PARAM_NAMES
        return {
            cls.LOC: 'mu',
            cls.SCALE: 'sigma',
            cls.SHAPE: 'zeta',
        }[param_name]

    def gumbel_standardization(self, x):
        x -= self.location
        x /= self.scale
        if self.shape == 0:
            return x
        else:
            return np.log(1 + self.shape * x) / self.shape

    def gumbel_inverse_standardization(self, x):
        if self.shape == 0:
            x = x
        else:
            x = (np.exp(self.shape * x) - 1) / self.shape
        x *= self.scale
        x += self.location
        return x



    @property
    def bound(self):
        return self.location - (self.scale / self.shape)

    @property
    def density_upper_bound(self):
        if self.shape >= 0:
            return np.inf
        else:
            return self.bound

    @property
    def density_lower_bound(self):
        if self.shape <= 0:
            return np.inf
        else:
            return self.bound

    def return_level_plot_against_return_period(self, ax=None, color=None, linestyle=None, label=None, show=False,
                                                suffix_return_level_label=''):
        if ax is None:
            ax = plt.gca()
        # Plot return level against return period
        return_periods = list(range(2, 61))
        quantiles = self.get_return_level(return_periods)
        return_period_to_quantile = dict(zip(return_periods, quantiles))
        ax.vlines(50, 0, return_period_to_quantile[50])
        ax.plot(return_periods, quantiles, color=color, linestyle=linestyle, label=label)
        ax.set_xlabel('Return period')
        ax.legend()

        ax.set_xticks([10 * i for i in range(1, 7)])
        ax.set_ylabel('Return level {}'.format(suffix_return_level_label))
        plt.gca().set_ylim(bottom=0)
        if show:
            plt.show()

    def get_return_level(self, return_periods):
        return np.array([self.quantile(1 - 1 / return_period) for return_period in return_periods])
