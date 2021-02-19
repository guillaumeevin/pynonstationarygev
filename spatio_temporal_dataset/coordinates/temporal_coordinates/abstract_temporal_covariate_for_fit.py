import pandas as pd

from extreme_data.meteo_france_data.adamont_data.adamont_scenario import str_to_scenario
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


class AnomalyTemperatureTemporalCovariate(AbstractTemporalCovariateForFit):
    gcm_and_scenario_to_d = {}

    @classmethod
    def get_temporal_covariate(cls, row: pd.Series):
        year = row[AbstractCoordinates.COORDINATE_T]
        gcm = row[AbstractCoordinates.COORDINATE_GCM]
        scenario_str = row[AbstractCoordinates.COORDINATE_RCP]
        scenario = str_to_scenario(scenario_str)
        if (gcm, scenario) not in cls.gcm_and_scenario_to_d:
            d = year_to_global_mean_temp(gcm, scenario, anomaly=True)
            cls.gcm_and_scenario_to_d[(gcm, scenario)] = d
        d = cls.gcm_and_scenario_to_d[(gcm, scenario)]
        global_mean_temp = d[year]
        return global_mean_temp
