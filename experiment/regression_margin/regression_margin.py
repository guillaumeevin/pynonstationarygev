import random

import numpy as np

from extreme_estimator.estimator.full_estimator import FullEstimatorInASingleStepWithSmoothMargin
from extreme_estimator.extreme_models.margin_model.margin_function.linear_margin_function import LinearMarginFunction
from extreme_estimator.extreme_models.margin_model.smooth_margin_model import LinearShapeDim1MarginModel, \
    LinearAllParametersAllDimsMarginModel
from extreme_estimator.extreme_models.max_stable_model.max_stable_models import Smith
from extreme_estimator.gev_params import GevParams
from spatio_temporal_dataset.coordinates.unidimensional_coordinates.coordinates_1D import LinSpaceCoordinates
import matplotlib.pyplot as plt

from spatio_temporal_dataset.dataset.simulation_dataset import FullSimulatedDataset

nb_points = 5
nb_obs = 10
nb_estimator = 2
show = False

coordinates = LinSpaceCoordinates.from_nb_points(nb_points=nb_points)

########## GENERATING THE DATA #####################

# MarginModel Linear with respect to the shape (from 0.01 to 0.02)
params_sample = {
    (GevParams.GEV_SHAPE, 0): 0.2,
    (GevParams.GEV_SHAPE, 1): 0.05,
}
margin_model = LinearShapeDim1MarginModel(coordinates=coordinates, params_sample=params_sample)
max_stable_model = Smith()

# if show:
#     # Plot a realization from the maxima gev (i.e the maxima obtained just by simulating the marginal law)
#     for maxima_gev in np.transpose(dataset.maxima_gev):
#         plt.plot(coordinates.coordinates_values, maxima_gev)
#     plt.show()

######### FITTING A MODEL #################


axes = None
for _ in range(nb_estimator):
    # Data part
    dataset = FullSimulatedDataset.from_double_sampling(nb_obs=nb_obs, margin_model=margin_model,
                                                        coordinates=coordinates,
                                                        max_stable_model=max_stable_model)
    margin_function_sample = dataset.margin_model.margin_function_sample # type: LinearMarginFunction
    margin_function_sample.visualize(show=False, axes=axes)
    axes = margin_function_sample.visualization_axes

    # Estimation part
    margin_model_for_estimator = LinearAllParametersAllDimsMarginModel(coordinates)
    full_estimator = FullEstimatorInASingleStepWithSmoothMargin(dataset, margin_model_for_estimator, max_stable_model)
    full_estimator.fit()
    full_estimator.margin_function_fitted.visualize(axes=axes, show=False)
plt.show()

# Display all the margin on the same graph for comparison

# Plot the margin functions
# margin_model.margin_function_sample.visualize_2D()
