import pandas as pd

from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation import \
    CenteredScaledNormalization
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


def load_temporal_coordinates_and_dataset(maxima, years):
    df = pd.DataFrame({AbstractCoordinates.COORDINATE_T: years})
    df_maxima_gev = pd.DataFrame(maxima, index=df.index)
    observations = AbstractSpatioTemporalObservations(df_maxima_gev=df_maxima_gev)
    coordinates = AbstractTemporalCoordinates.from_df(df, transformation_class=CenteredScaledNormalization)
    dataset = AbstractDataset(observations=observations, coordinates=coordinates)
    return coordinates, dataset


def fitted_linear_margin_estimator(model_class, coordinates, dataset, starting_year, fit_method, **model_kwargs):
    model = model_class(coordinates, starting_point=starting_year, fit_method=fit_method, **model_kwargs)
    estimator = LinearMarginEstimator(dataset, model)
    estimator.fit()
    return estimator
