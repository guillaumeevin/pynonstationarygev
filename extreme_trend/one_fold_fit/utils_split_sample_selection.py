import math

import numpy as np

from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.estimator.margin_estimator.utils import fitted_linear_margin_estimator_short
from extreme_fit.estimator.margin_estimator.utils_functions import compute_nllh, NllhIsInfException
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


def compute_mean_log_score_with_split_sample(global_estimator: LinearMarginEstimator):
    ratio = 0.5
    nb_train_obs = math.ceil(61 * ratio)
    nb_test_obs = 61 - nb_train_obs
    # Load splits
    global_dataset = global_estimator.dataset
    global_coordinates = global_dataset.coordinates
    global_df_coordinates = global_coordinates.df_coordinates(add_climate_informations=True)
    gcm_rcm_couple_as_pseudo_truth = global_coordinates.gcm_rcm_couple_as_pseudo_truth
    if gcm_rcm_couple_as_pseudo_truth is None:
        ind_test_split = global_df_coordinates[AbstractCoordinates.COORDINATE_RCM].isnull()
    else:
        gcm, rcm = gcm_rcm_couple_as_pseudo_truth
        ind_test_split = (global_df_coordinates[AbstractCoordinates.COORDINATE_RCM] == rcm) \
                         & (global_df_coordinates[AbstractCoordinates.COORDINATE_GCM] == gcm)
    t_coordinates = global_df_coordinates.loc[ind_test_split, AbstractCoordinates.COORDINATE_T].values
    index_to_start_test = math.ceil(len(t_coordinates) * ratio)
    value_to_start_test = sorted(t_coordinates)[index_to_start_test]
    ind_test_split &= global_df_coordinates.loc[ind_test_split, AbstractCoordinates.COORDINATE_T] >= value_to_start_test
    assert sum(ind_test_split) == nb_test_obs
    ind_train_split = ~ind_test_split
    coordinates = AbstractCoordinates.from_df(global_df_coordinates.loc[ind_train_split].copy())
    coordinates.gcm_rcm_couple_as_pseudo_truth = global_coordinates.gcm_rcm_couple_as_pseudo_truth
    observations = AbstractSpatioTemporalObservations(
        df_maxima_gev=global_dataset.observations.df_maxima_gev.loc[ind_train_split].copy())
    dataset = AbstractDataset(observations, coordinates)
    # Fit small estimator
    local_estimator = fitted_linear_margin_estimator_short(model_class=type(global_estimator.margin_model),
                                                           dataset=dataset,
                                                           fit_method=global_estimator.margin_model.fit_method,
                                                           temporal_covariate_for_fit=global_estimator.margin_model.temporal_covariate_for_fit,
                                                           drop_duplicates=False,
                                                           param_name_to_climate_coordinates_with_effects=global_estimator.margin_model.param_name_to_climate_coordinates_with_effects)
    # Load test dataset after the split

    maxima_values = global_dataset.observations.df_maxima_gev.loc[ind_test_split].values
    coordinate_values = global_estimator.df_coordinates_temp.loc[ind_test_split].values
    # Compute the list of log score
    try:
        mean_log_score = compute_nllh(coordinate_values, maxima_values,
                                      local_estimator.margin_function_from_fit) / nb_test_obs
    except NllhIsInfException:
        mean_log_score = np.Inf
    return mean_log_score
