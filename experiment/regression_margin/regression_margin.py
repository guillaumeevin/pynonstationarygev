import numpy as np

from extreme_estimator.estimator.full_estimator import FullEstimatorInASingleStepWithSmoothMargin
from extreme_estimator.extreme_models.margin_model.smooth_margin_model import LinearShapeAxis0MarginModel, \
    LinearAllParametersAllAxisMarginModel
from extreme_estimator.extreme_models.max_stable_model.max_stable_models import Smith
from extreme_estimator.gev_params import GevParams
from spatio_temporal_dataset.coordinates.unidimensional_coordinates.coordinates_1D import LinSpaceCoordinates
import matplotlib.pyplot as plt

from spatio_temporal_dataset.dataset.simulation_dataset import FullSimulatedDataset

nb_points = 50
nb_obs = 100

coordinates = LinSpaceCoordinates.from_nb_points(nb_points=nb_points)


########## GENERATING THE DATA #####################

# MarginModel Linear with respect to the shape (from 0.01 to 0.02)
margin_model = LinearShapeAxis0MarginModel(coordinates=coordinates, params_sample={GevParams.GEV_SHAPE: 0.02})
max_stable_model = Smith()
dataset = FullSimulatedDataset.from_double_sampling(nb_obs=nb_obs, margin_model=margin_model,
                                                    coordinates=coordinates,
                                                    max_stable_model=max_stable_model)
# Visualize the sampling margin
dataset.margin_model.margin_function_sample.visualize_all()
# Plot a realization from the maxima gev (i.e the maxima obtained just by simulating the marginal law)
for maxima_gev in np.transpose(dataset.maxima_gev):
    plt.plot(coordinates.coordinates_values, maxima_gev)
plt.show()

######### FITTING A MODEL #################

margin_model = LinearAllParametersAllAxisMarginModel(coordinates)
full_estimator = FullEstimatorInASingleStepWithSmoothMargin(dataset, margin_model, max_stable_model)
full_estimator.fit()
print(full_estimator.full_params_fitted)

# Plot the margin functions
# margin_model.margin_function_sample.visualize_2D()
