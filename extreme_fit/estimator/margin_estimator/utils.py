import warnings

import numpy as np
import pandas as pd

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation import \
    CenteredScaledNormalization
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import AnnualMaxima


def fitted_linear_margin_estimator_short(model_class, dataset, fit_method, drop_duplicates=None,
                                         param_name_to_climate_coordinates_with_effects=None,
                                         linear_effects=(False, False, False),
                                         **model_kwargs) -> LinearMarginEstimator:
    return fitted_linear_margin_estimator(model_class, dataset.coordinates, dataset, None,
                                          fit_method, drop_duplicates, param_name_to_climate_coordinates_with_effects,
                                          linear_effects,
                                          **model_kwargs)


def fitted_linear_margin_estimator(model_class, coordinates, dataset, starting_year, fit_method, drop_duplicates=None,
                                   param_name_to_climate_coordinates_with_effects=None, linear_effects=(False, False, False),
                                   **model_kwargs):
    model = model_class(coordinates, starting_point=starting_year,
                        fit_method=fit_method, param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects,
                        linear_effects=linear_effects,
                        **model_kwargs)
    if drop_duplicates is not None:
        model.drop_duplicates = drop_duplicates
    estimator = LinearMarginEstimator(dataset, model)
    estimator.fit()
    return estimator


def fitted_stationary_gev(x_gev, fit_method=MarginFitMethod.is_mev_gev_fit, model_class=StationaryTemporalModel,
                          starting_year=None,
                          transformation_class=CenteredScaledNormalization) -> GevParams:
    _, gev_param = _fitted_stationary_gev(fit_method, model_class, starting_year, x_gev)
    return gev_param


def _fitted_stationary_gev(fit_method, model_class, starting_year, x_gev):
    coordinates = ConsecutiveTemporalCoordinates.from_nb_temporal_steps(nb_temporal_steps=len(x_gev),
                                                                        transformation_class=CenteredScaledNormalization)
    df = pd.DataFrame([pd.Series(dict(zip(coordinates.index, x_gev)))]).transpose()
    observations = AnnualMaxima(df_maxima_gev=df)
    dataset = AbstractDataset(observations=observations, coordinates=coordinates)
    estimator = fitted_linear_margin_estimator(model_class, coordinates, dataset, starting_year, fit_method)
    first_coordinate = coordinates.coordinates_values()[0]
    gev_param = estimator.margin_function_from_fit.get_params(first_coordinate)
    # Warning
    if not -0.5 < gev_param.shape < 0.5:
        pass
        # warnings.warn('fitted shape parameter is outside physical bounds {}'.format(gev_param.shape))
    return estimator, gev_param
