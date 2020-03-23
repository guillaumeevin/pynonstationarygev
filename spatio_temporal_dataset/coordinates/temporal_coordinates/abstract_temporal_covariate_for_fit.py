import pandas as pd

from extreme_data.nasa_data.global_mean_temperature import load_year_to_mean_global_temperature
from root_utils import classproperty


class AbstractTemporalCovariateForFit(object):

    @classmethod
    def get_temporal_covariate(cls, t):
        raise NotImplementedError


class TimeTemporalCovariate(AbstractTemporalCovariateForFit):

    @classmethod
    def get_temporal_covariate(cls, t):
        return t


class MeanGlobalTemperatureCovariate(AbstractTemporalCovariateForFit):

    _d = None

    @classproperty
    def year_to_global_mean(cls):
        if cls._d is None:
            cls._d = load_year_to_mean_global_temperature()
        return cls._d

    @classmethod
    def get_temporal_covariate(cls, t):
        try:
            return pd.Series(cls.year_to_global_mean[t])
        except KeyError:
            raise KeyError('Global mean temperature is not known for Year t={}'.format(t))
