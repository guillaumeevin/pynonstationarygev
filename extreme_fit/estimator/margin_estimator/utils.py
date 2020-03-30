import pandas as pd

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation import \
    CenteredScaledNormalization
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import AnnualMaxima


def fitted_linear_margin_estimator(model_class, coordinates, dataset, starting_year, fit_method, **model_kwargs):
    model = model_class(coordinates, starting_point=starting_year, fit_method=fit_method, **model_kwargs)
    estimator = LinearMarginEstimator(dataset, model)
    estimator.fit()
    return estimator


def fitted_stationary_gev(x_gev, fit_method, model_class=StationaryTemporalModel, starting_year=None,
                          transformation_class=CenteredScaledNormalization) -> GevParams:
    coordinates = ConsecutiveTemporalCoordinates.from_nb_temporal_steps(nb_temporal_steps=len(x_gev),
                                                                        transformation_class=CenteredScaledNormalization)
    df = pd.DataFrame([pd.Series(dict(zip(coordinates.index, x_gev)))]).transpose()
    observations = AnnualMaxima(df_maxima_gev=df)
    dataset = AbstractDataset(observations=observations, coordinates=coordinates)
    estimator = fitted_linear_margin_estimator(model_class, coordinates, dataset, starting_year, fit_method)
    first_coordinate = coordinates.coordinates_values()[0]
    gev_param = estimator.function_from_fit.get_gev_params(first_coordinate)
    return gev_param
