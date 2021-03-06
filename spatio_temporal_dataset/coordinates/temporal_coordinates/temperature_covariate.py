from extreme_data.cru_data.global_mean_temperature_until_2020 import \
    winter_year_to_averaged_global_mean_temp_wrt_1850_1900
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import str_to_scenario, get_gcm_list, AdamontScenario
from extreme_data.meteo_france_data.adamont_data.cmip5.climate_explorer_cimp5 import year_to_global_mean_temp, \
    year_to_averaged_global_mean_temp
from extreme_data.meteo_france_data.mean_alps_temperature import load_year_to_mean_alps_temperatures
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

class MeanAlpsTemperatureCovariate(AbstractTemperatureCovariate):

    @classmethod
    def load_year_to_temperature_covariate(cls):
        return load_year_to_mean_alps_temperatures()


class AnomalyTemperatureWithSplineTemporalCovariate(AbstractTemporalCovariateForFit):
    gcm_and_scenario_to_d = {}

    @classmethod
    def get_temporal_covariate(cls, row: pd.Series):
        year = row[AbstractCoordinates.COORDINATE_T]
        gcm = row[AbstractCoordinates.COORDINATE_GCM]
        scenario_str = row[AbstractCoordinates.COORDINATE_RCP]
        scenario = str_to_scenario(scenario_str) if isinstance(scenario_str, str) else None
        is_observations = isinstance(gcm, float) and isinstance(scenario_str, float)
        if (gcm, scenario) not in cls.gcm_and_scenario_to_d:
            if is_observations:
                d = winter_year_to_averaged_global_mean_temp_wrt_1850_1900(spline=True)
            else:
                d = year_to_global_mean_temp(gcm, scenario, anomaly=True, spline=True)
            cls.gcm_and_scenario_to_d[(gcm, scenario)] = d
        d = cls.gcm_and_scenario_to_d[(gcm, scenario)]
        global_mean_temp = d[year]
        return global_mean_temp
