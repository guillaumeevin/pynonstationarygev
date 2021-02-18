from extreme_data.meteo_france_data.mean_alps_temperature import load_year_to_mean_alps_temperatures
from extreme_data.nasa_data.global_mean_temperature import load_year_to_mean_global_temperature
from root_utils import classproperty
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    AbstractTemporalCovariateForFit
import pandas as pd


class AbstractTemperatureCovariate(AbstractTemporalCovariateForFit):
    _d = None

    @classproperty
    def year_to_global_mean(cls):
        if cls._d is None:
            cls._d = cls.load_year_to_temperature_covariate()
        return cls._d

    @classmethod
    def load_year_to_temperature_covariate(cls):
        raise NotImplemented

    @classmethod
    def get_temporal_covariate(cls, row):
        t = row[AbstractCoordinates.COORDINATE_T]
        try:
            return pd.Series(cls.year_to_global_mean[t])
        except KeyError:
            raise KeyError('Global mean temperature is not known for Year t={}'.format(t))


class MeanGlobalTemperatureCovariate(AbstractTemperatureCovariate):

    @classmethod
    def load_year_to_temperature_covariate(cls):
        return load_year_to_mean_global_temperature()


class MeanAlpsTemperatureCovariate(AbstractTemperatureCovariate):

    @classmethod
    def load_year_to_temperature_covariate(cls):
        return load_year_to_mean_alps_temperatures()
