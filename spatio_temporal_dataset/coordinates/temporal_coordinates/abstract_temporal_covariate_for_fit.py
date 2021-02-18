import pandas as pd

from extreme_data.meteo_france_data.adamont_data.cmip5.climate_explorer_cimp5 import year_to_global_mean_temp
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractTemporalCovariateForFit(object):

    @classmethod
    def get_temporal_covariate(cls, row: pd.Series):
        raise NotImplementedError


class TimeTemporalCovariate(AbstractTemporalCovariateForFit):

    @classmethod
    def get_temporal_covariate(cls, row: pd.Series):
        return row[AbstractCoordinates.COORDINATE_T]


class TemperatureTemporalCovariate(AbstractTemporalCovariateForFit):
    gcm_and_scenario_to_d = {}

    @classmethod
    def get_temporal_covariate(cls, row: pd.Series):
        year = row[AbstractCoordinates.COORDINATE_T]
        gcm = None
        scenario = None
        if (gcm, scenario) not in cls.gcm_and_scenario_to_d:
            d = year_to_global_mean_temp(gcm, scenario)
            cls.gcm_and_scenario_to_d[(gcm, scenario)] = d
        d = cls.gcm_and_scenario_to_d[(gcm, scenario)]
        global_mean_temp = d[year]
        print(type(global_mean_temp))
        return global_mean_temp
