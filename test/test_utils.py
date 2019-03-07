from itertools import product
from typing import List

from experiment.meteo_france_SCM_study.abstract_study import AbstractStudy
from experiment.meteo_france_SCM_study.crocus.crocus import Crocus, CrocusSwe, CrocusDepth
from experiment.meteo_france_SCM_study.crocus.crocus_variables import CrocusSweVariable, CrocusDepthVariable
from extreme_estimator.estimator.full_estimator.abstract_full_estimator import SmoothMarginalsThenUnitaryMsp, \
    FullEstimatorInASingleStepWithSmoothMargin
from extreme_estimator.estimator.max_stable_estimator.abstract_max_stable_estimator import MaxStableEstimator
from extreme_estimator.extreme_models.margin_model.smooth_margin_model import LinearAllParametersAllDimsMarginModel, \
    ConstantMarginModel
from extreme_estimator.extreme_models.max_stable_model.abstract_max_stable_model import \
    AbstractMaxStableModelWithCovarianceFunction, CovarianceFunction
from extreme_estimator.extreme_models.max_stable_model.max_stable_models import Smith, BrownResnick, Schlather, \
    Geometric, ExtremalT, ISchlather
from experiment.meteo_france_SCM_study.safran.safran import SafranSnowfall, Safran
from spatio_temporal_dataset.coordinates.spatial_coordinates.alps_station_3D_coordinates import \
    AlpsStation3DCoordinatesWithAnisotropy
from spatio_temporal_dataset.coordinates.spatial_coordinates.generated_spatial_coordinates import \
    CircleSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.generated_spatio_temporal_coordinates import \
    UniformSpatioTemporalCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import UniformSpatialCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates

"""
Common objects to load for the test.
Sometimes it doesn't cover all the class (e.g margin_model, coordinates...)
In this case, unit test (at least on the constructor) must be ensured in the test relative to the class 
"""

TEST_MAX_STABLE_MODEL = [Smith, BrownResnick, Schlather, Geometric, ExtremalT, ISchlather]
TEST_1D_AND_2D_SPATIAL_COORDINATES = [UniformSpatialCoordinates, CircleSpatialCoordinates]
TEST_3D_SPATIAL_COORDINATES = [AlpsStation3DCoordinatesWithAnisotropy]
TEST_TEMPORAL_COORDINATES = [ConsecutiveTemporalCoordinates]
TEST_SPATIO_TEMPORAL_COORDINATES = [UniformSpatioTemporalCoordinates]
TEST_MARGIN_TYPES = [ConstantMarginModel, LinearAllParametersAllDimsMarginModel][:]
TEST_MAX_STABLE_ESTIMATOR = [MaxStableEstimator]
TEST_FULL_ESTIMATORS = [SmoothMarginalsThenUnitaryMsp, FullEstimatorInASingleStepWithSmoothMargin][:]


def load_test_full_estimators(dataset, margin_model, max_stable_model):
    return [full_estimator(dataset=dataset, margin_model=margin_model, max_stable_model=max_stable_model) for
            full_estimator in TEST_FULL_ESTIMATORS]


def load_test_max_stable_estimators(dataset, max_stable_model):
    return [max_stable_estimator(dataset, max_stable_model) for max_stable_estimator in TEST_MAX_STABLE_ESTIMATOR]


def load_smooth_margin_models(coordinates):
    return [margin_class(coordinates=coordinates) for margin_class in TEST_MARGIN_TYPES]


def load_test_max_stable_models(default_covariance_function=None):
    # Load all max stable model
    max_stable_models = []
    for max_stable_class in TEST_MAX_STABLE_MODEL:
        if issubclass(max_stable_class, AbstractMaxStableModelWithCovarianceFunction):
            if default_covariance_function is not None:
                assert default_covariance_function in CovarianceFunction
                max_stable_models.append(max_stable_class(covariance_function=default_covariance_function))
            else:
                max_stable_models.extend([max_stable_class(covariance_function=covariance_function)
                                          for covariance_function in CovarianceFunction])
        else:
            max_stable_models.append(max_stable_class())
    return max_stable_models


def load_test_spatial_coordinates(nb_points, coordinate_types, train_split_ratio=None):
    return [coordinate_class.from_nb_points(nb_points=nb_points, train_split_ratio=train_split_ratio)
            for coordinate_class in coordinate_types]


def load_test_1D_and_2D_spatial_coordinates(nb_points, train_split_ratio=None):
    return load_test_spatial_coordinates(nb_points, TEST_1D_AND_2D_SPATIAL_COORDINATES,
                                         train_split_ratio=train_split_ratio)


def load_test_3D_spatial_coordinates(nb_points):
    return load_test_spatial_coordinates(nb_points, TEST_3D_SPATIAL_COORDINATES)


def load_test_temporal_coordinates(nb_steps, train_split_ratio=None):
    return [coordinate_class.from_nb_temporal_steps(nb_steps, train_split_ratio) for coordinate_class in
            TEST_TEMPORAL_COORDINATES]


def load_test_spatiotemporal_coordinates(nb_points, nb_steps, train_split_ratio=None):
    return [coordinate_class.from_nb_points_and_nb_steps(nb_points=nb_points, nb_steps=nb_steps,
                                                         train_split_ratio=train_split_ratio)
            for coordinate_class in TEST_SPATIO_TEMPORAL_COORDINATES]


def load_safran_studies(altitudes) -> List[Safran]:
    nb_days_list = [1]
    return [SafranSnowfall(safran_altitude, nb_days) for safran_altitude in altitudes for nb_days in nb_days_list]


def load_crocus_studies(altitudes) -> List[Crocus]:
    crocus_classes = [CrocusSwe, CrocusDepth][:]
    return [crocus_class(altitude) for crocus_class, altitude in product(crocus_classes, altitudes)]


def load_scm_studies() -> List[AbstractStudy]:
    altitudes = [1800, 2400][:]
    scm_studies = load_safran_studies(altitudes)
    scm_studies += load_crocus_studies(altitudes)
    return scm_studies
