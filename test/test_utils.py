from extreme_estimator.extreme_models.margin_model.smooth_margin_model import LinearAllParametersAxis0And1MarginModel, \
    ConstantMarginModel
from extreme_estimator.extreme_models.max_stable_model.abstract_max_stable_model import \
    AbstractMaxStableModelWithCovarianceFunction, CovarianceFunction
from extreme_estimator.extreme_models.max_stable_model.max_stable_models import Smith, BrownResnick, Schlather, \
    Geometric, ExtremalT, ISchlather
from spatio_temporal_dataset.coordinates.spatial_coordinates.alps_station_3D_coordinates import \
    AlpsStation3DCoordinatesWithAnisotropy
from spatio_temporal_dataset.coordinates.spatial_coordinates.generated_spatial_coordinates import CircleCoordinates
from spatio_temporal_dataset.coordinates.unidimensional_coordinates.coordinates_1D import UniformCoordinates

"""
Common objects to load for the test.
Sometimes it doesn't cover all the class (e.g margin_model, coordinates...)
In this case, unit test (at least on the constructor) must be ensured in the test relative to the class 
"""

TEST_MAX_STABLE_MODEL = [Smith, BrownResnick, Schlather, Geometric, ExtremalT, ISchlather]
TEST_COORDINATES = [UniformCoordinates, CircleCoordinates, AlpsStation3DCoordinatesWithAnisotropy]
MARGIN_TYPES = [ConstantMarginModel, LinearAllParametersAxis0And1MarginModel][:]


def load_smooth_margin_models(coordinates):
    return [margin_class(coordinates=coordinates) for margin_class in MARGIN_TYPES]


def load_test_max_stable_models():
    # Load all max stable model
    max_stable_models = []
    for max_stable_class in TEST_MAX_STABLE_MODEL:
        if issubclass(max_stable_class, AbstractMaxStableModelWithCovarianceFunction):
            max_stable_models.extend([max_stable_class(covariance_function=covariance_function)
                                      for covariance_function in CovarianceFunction])
        else:
            max_stable_models.append(max_stable_class())
    return max_stable_models


def load_test_coordinates(nb_points):
    return [coordinate_class.from_nb_points(nb_points=nb_points) for coordinate_class in TEST_COORDINATES]
