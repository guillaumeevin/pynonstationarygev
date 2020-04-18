from collections import OrderedDict
from typing import List, Dict

import numpy as np
from cached_property import cached_property

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.estimator.quantile_estimator.abstract_quantile_estimator import AbstractQuantileEstimator
from extreme_fit.model.margin_model.abstract_margin_model import AbstractMarginModel
from extreme_fit.model.margin_model.utils import \
    MarginFitMethod
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    NonStationaryLocationTemporalModel
from projects.quantile_regression_vs_evt.AbstractSimulation import AbstractSimulation
from projects.quantile_regression_vs_evt.annual_maxima_simulation.abstract_annual_maxima_simulation import \
    AnnualMaximaSimulation
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import MarginAnnualMaxima




class GevSimulation(AnnualMaximaSimulation):

    @property
    def observations_class(self):
        return MarginAnnualMaxima

class StationarySimulation(GevSimulation):

    def create_model(self, coordinates):
        param_name_to_coef_list = {
            GevParams.LOC: [0],
            GevParams.SHAPE: [0],
            GevParams.SCALE: [1],
        }
        return StationaryTemporalModel.from_coef_list(coordinates, param_name_to_coef_list,
                                                      fit_method=MarginFitMethod.extremes_fevd_mle)


class NonStationaryLocationGumbelSimulation(GevSimulation):

    def create_model(self, coordinates):
        param_name_to_coef_list = {
            GevParams.LOC: [0, 10],
            GevParams.SHAPE: [0],
            GevParams.SCALE: [1],
        }
        return NonStationaryLocationTemporalModel.from_coef_list(coordinates, param_name_to_coef_list,
                                                                 fit_method=MarginFitMethod.extremes_fevd_mle)


class NonStationaryLocationGevSimulation(GevSimulation):

    def create_model(self, coordinates):
        param_name_to_coef_list = {
            GevParams.LOC: [0, 1],
            GevParams.SHAPE: [0.3],
            GevParams.SCALE: [1],
        }
        return NonStationaryLocationTemporalModel.from_coef_list(coordinates, param_name_to_coef_list,
                                                                 fit_method=MarginFitMethod.extremes_fevd_mle)
