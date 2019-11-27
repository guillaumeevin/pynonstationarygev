from collections import OrderedDict

import numpy as np
from cached_property import cached_property

from experiment.meteo_france_data.scm_models_data.crocus.crocus import Crocus, CrocusSweTotal, CrocusSnowLoadTotal, \
    CrocusSnowLoadEurocode, CrocusDepth
from experiment.meteo_france_data.scm_models_data.crocus.crocus_variables import TotalSnowLoadVariable, \
    CrocusDensityVariable, CrocusDepthVariable


class CrocusSnowDepthAtMaxofSwe(Crocus):

    def __init__(self, *args, **kwargs):
        super().__init__(CrocusDepthVariable, *args, **kwargs)
        self.year_to_snow_depth_at_max_of_swe = OrderedDict()
        study_swe = CrocusSweTotal(*args, **kwargs)
        for year in study_swe.ordered_years:
            argmax_swe = study_swe.year_to_annual_maxima_index[year]
            snow_depth = self.year_to_daily_time_serie_array[year]
            snow_depth_at_max = [snow_depth[argmax, i] for i, argmax in enumerate(argmax_swe)]
            self.year_to_snow_depth_at_max_of_swe[year] = np.array(snow_depth_at_max)

    @cached_property
    def year_to_annual_maxima(self) -> OrderedDict:
        return self.year_to_snow_depth_at_max_of_swe


class CrocusSnowDepthDifference(Crocus):

    def __init__(self, *args, **kwargs):
        super().__init__(CrocusDepthVariable, *args, **kwargs)
        snow_depth_at_max_swe = CrocusSnowDepthAtMaxofSwe(*args, **kwargs)
        snow_depth_max = CrocusDepth(*args, **kwargs)
        self.year_to_diff = OrderedDict()
        for year in snow_depth_max.ordered_years:
            self.year_to_diff[year] = snow_depth_max.year_to_annual_maxima[year] \
                                      - snow_depth_at_max_swe.year_to_annual_maxima[year]

    @property
    def year_to_annual_maxima(self) -> OrderedDict:
        return self.year_to_diff


class CrocusSnowDensityAtMaxofSwe(Crocus):

    def __init__(self, *args, **kwargs):
        super().__init__(CrocusDensityVariable, *args, **kwargs)
        study_swe = CrocusSweTotal(*args, **kwargs)
        self.year_to_snow_density_at_max_of_swe = OrderedDict()
        for year in study_swe.ordered_years:
            max_swe = study_swe.year_to_annual_maxima[year]
            argmax_swe = study_swe.year_to_annual_maxima_index[year]
            snow_depth = self.year_to_daily_time_serie_array[year]
            snow_depth_at_max = np.array([snow_depth[argmax, i] for i, argmax in enumerate(argmax_swe)])
            # todo: should i take into acocunt the density of the water
            self.year_to_snow_density_at_max_of_swe[year] = max_swe / snow_depth_at_max

    @property
    def year_to_annual_maxima(self) -> OrderedDict:
        return self.year_to_snow_density_at_max_of_swe


class CrocusDifferenceSnowLoadRescaledAndEurocodeToSeeSynchronization(Crocus):

    def __init__(self, *args, **kwargs):
        super().__init__(TotalSnowLoadVariable, *args, **kwargs)
        study_snow_load = CrocusSnowLoadTotal(*args, **kwargs)
        study_snow_load_eurocode = CrocusSnowLoadEurocode(*args, **kwargs)
        study_density = CrocusSnowDensityAtMaxofSwe(*args, **kwargs)
        self.year_to_snow_difference = OrderedDict()
        for year in study_density.ordered_years:
            rescaling_factor = study_density.year_to_annual_maxima[year] / 150
            snow_load_rescaled = study_snow_load.year_to_annual_maxima[year] / rescaling_factor
            self.year_to_snow_difference[year] = snow_load_rescaled - study_snow_load_eurocode.year_to_annual_maxima[
                year]

    @property
    def year_to_annual_maxima(self) -> OrderedDict:
        return self.year_to_snow_difference


class CrocusDifferenceSnowLoad(Crocus):

    def __init__(self, *args, **kwargs):
        super().__init__(TotalSnowLoadVariable, *args, **kwargs)
        study_snow_load = CrocusSnowLoadTotal(*args, **kwargs)
        study_snow_load_eurocode = CrocusSnowLoadEurocode(*args, **kwargs)
        self.year_to_snow_difference = OrderedDict()
        for year in study_snow_load.ordered_years:
            self.year_to_snow_difference[year] = study_snow_load.year_to_annual_maxima[year] \
                                                 - study_snow_load_eurocode.year_to_annual_maxima[year]

    @property
    def year_to_annual_maxima(self) -> OrderedDict:
        return self.year_to_snow_difference