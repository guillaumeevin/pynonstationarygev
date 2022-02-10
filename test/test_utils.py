from itertools import product
from typing import List

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import Crocus, CrocusSweTotal, CrocusDepth
from extreme_fit.estimator.full_estimator.abstract_full_estimator import SmoothMarginalsThenUnitaryMsp, \
    FullEstimatorInASingleStepWithSmoothMargin
from extreme_fit.estimator.max_stable_estimator.abstract_max_stable_estimator import MaxStableEstimator
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import \
    LinearAllParametersAllDimsMarginModel, \
    ConstantMarginModel
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationTemporalModel, NonStationaryScaleTemporalModel, NonStationaryShapeTemporalModel
from extreme_fit.model.max_stable_model.abstract_max_stable_model import \
    AbstractMaxStableModelWithCovarianceFunction, CovarianceFunction
from extreme_fit.model.max_stable_model.max_stable_models import Smith, BrownResnick, Schlather, \
    Geometric, ExtremalT, ISchlather
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall, Safran, SafranRainfall, \
    SafranTemperature, SafranPrecipitation
from extreme_fit.model.quantile_model.quantile_regression_model import ConstantQuantileRegressionModel, \
    TemporalCoordinatesQuantileRegressionModel
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.generated_spatial_coordinates import \
    CircleSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.generated_spatio_temporal_coordinates import \
    UniformSpatioTemporalCoordinates, LinSpaceSpatial2DSpatioTemporalCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import UniformSpatialCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates

"""
Common objects to load for the test.
Sometimes it doesn't cover all the class (e.g margin_model, coordinates...)
In this case, unit test (at least on the constructor) must be ensured in the test relative to the class 
"""

TEST_MAX_STABLE_MODEL = [Smith, BrownResnick, Schlather, Geometric, ExtremalT, ISchlather]
TEST_1D_AND_2D_SPATIAL_COORDINATES = [UniformSpatialCoordinates, CircleSpatialCoordinates]
TEST_TEMPORAL_COORDINATES = [ConsecutiveTemporalCoordinates]
TEST_SPATIO_TEMPORAL_COORDINATES = [UniformSpatioTemporalCoordinates, LinSpaceSpatial2DSpatioTemporalCoordinates]
TEST_MARGIN_TYPES = [ConstantMarginModel, LinearAllParametersAllDimsMarginModel][:]
TEST_QUANTILES_TYPES = [ConstantQuantileRegressionModel, TemporalCoordinatesQuantileRegressionModel][:]
TEST_NON_STATIONARY_TEMPORAL_MARGIN_TYPES = [NonStationaryLocationTemporalModel, NonStationaryScaleTemporalModel,
                                             NonStationaryShapeTemporalModel]
TEST_MAX_STABLE_ESTIMATOR = [MaxStableEstimator]
TEST_FULL_ESTIMATORS = [SmoothMarginalsThenUnitaryMsp, FullEstimatorInASingleStepWithSmoothMargin][:]


def load_non_stationary_temporal_margin_models(coordinates):
    return [margin_class(coordinates=coordinates) for margin_class in TEST_NON_STATIONARY_TEMPORAL_MARGIN_TYPES]


def load_test_full_estimators(dataset, margin_model, max_stable_model):
    return [full_estimator(dataset=dataset, margin_model=margin_model, max_stable_model=max_stable_model) for
            full_estimator in TEST_FULL_ESTIMATORS]


def load_test_max_stable_estimators(dataset, max_stable_model):
    return [max_stable_estimator(dataset, max_stable_model) for max_stable_estimator in TEST_MAX_STABLE_ESTIMATOR]


def load_smooth_margin_models(coordinates):
    return [margin_class(coordinates=coordinates) for margin_class in TEST_MARGIN_TYPES]


def load_smooth_quantile_model_classes():
    return [quantile_reg_class for quantile_reg_class in TEST_QUANTILES_TYPES]


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


def load_test_spatial_coordinates(nb_points, coordinate_types, transformation_class=None):
    return [coordinate_class.from_nb_points(nb_points=nb_points,
                                            transformation_class=transformation_class)
            for coordinate_class in coordinate_types]


def load_test_1D_and_2D_spatial_coordinates(nb_points, transformation_class=None) -> List[
    AbstractSpatialCoordinates]:
    return load_test_spatial_coordinates(nb_points, TEST_1D_AND_2D_SPATIAL_COORDINATES,
                                         transformation_class=transformation_class)

def load_test_temporal_coordinates(nb_steps, transformation_class=None) -> List[AbstractTemporalCoordinates]:
    return [coordinate_class.from_nb_temporal_steps(nb_temporal_steps=nb_steps,
                                                    transformation_class=transformation_class)
            for coordinate_class in TEST_TEMPORAL_COORDINATES]


def load_test_spatiotemporal_coordinates(nb_points, nb_steps, transformation_class: type = None):
    return [coordinate_class.from_nb_points_and_nb_steps(nb_points=nb_points, nb_steps=nb_steps,
                                                         transformation_class=transformation_class)
            for coordinate_class in TEST_SPATIO_TEMPORAL_COORDINATES]


def load_safran_studies(altitudes) -> List[Safran]:
    nb_days_list = [1]
    safran_studies = [safran_class(altitude=safran_altitude, nb_consecutive_days=nb_days)
                      for safran_altitude in altitudes for nb_days in nb_days_list
                      for safran_class in [SafranSnowfall, SafranRainfall, SafranPrecipitation]]
    safran_studies += [SafranTemperature(altitude) for altitude in altitudes]
    return safran_studies


def load_crocus_studies(altitudes) -> List[Crocus]:
    crocus_classes = [CrocusSweTotal, CrocusDepth][:]
    return [crocus_class(altitude=altitude) for crocus_class, altitude in product(crocus_classes, altitudes)]


def load_scm_studies() -> List[AbstractStudy]:
    altitudes = [1800, 2400][:]
    scm_studies = load_safran_studies(altitudes)
    scm_studies += load_crocus_studies(altitudes)
    return scm_studies
