import numpy as np

from extreme_estimator.estimator.full_estimator.abstract_full_estimator import FullEstimatorInASingleStepWithSmoothMargin
from extreme_estimator.extreme_models.margin_model.margin_function.linear_margin_function import LinearMarginFunction
from extreme_estimator.extreme_models.margin_model.linear_margin_model.linear_margin_model import LinearAllParametersAllDimsMarginModel, \
    ConstantMarginModel
from extreme_estimator.extreme_models.max_stable_model.max_stable_models import Smith
from extreme_estimator.margin_fits.gev.gev_params import GevParams
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import LinSpaceSpatialCoordinates
import matplotlib.pyplot as plt

from spatio_temporal_dataset.dataset.simulation_dataset import FullSimulatedDataset

nb_points = 50
nb_obs = 60
nb_estimator = 2
show = False

coordinates = LinSpaceSpatialCoordinates.from_nb_points(nb_points=nb_points)

########## GENERATING THE DATA #####################

# MarginModel Linear with respect to the shape (from 0.01 to 0.02)
params_sample = {
    # (GevParams.GEV_SHAPE, 0): 0.2,
    (GevParams.LOC, 0): 10,
    (GevParams.SHAPE, 0): 1.0,
    (GevParams.SCALE, 0): 1.0,
}
margin_model = ConstantMarginModel(coordinates=coordinates, params_sample=params_sample)
margin_model_for_estimator_class = [LinearAllParametersAllDimsMarginModel, ConstantMarginModel][-1]
max_stable_model = Smith()


######### FITTING A MODEL #################


axes = None
for i in range(nb_estimator):
    print("{}/{}".format(i+1, nb_estimator))
    # Data part
    dataset = FullSimulatedDataset.from_double_sampling(nb_obs=nb_obs, margin_model=margin_model,
                                                        coordinates=coordinates,
                                                        max_stable_model=max_stable_model)

    if show and i == 0:
        # Plot a realization from the maxima margin_fits (i.e the maxima obtained just by simulating the marginal law)
        for maxima in np.transpose(dataset.maxima_frech()):
            plt.plot(coordinates.coordinates_values(), maxima, 'o')
        plt.show()

    margin_function_sample = dataset.margin_model.margin_function_sample # type: LinearMarginFunction
    margin_function_sample.visualize_function(show=False, axes=axes, dot_display=True)
    axes = margin_function_sample.visualization_axes

    # Estimation part
    margin_model_for_estimator = margin_model_for_estimator_class(coordinates)
    full_estimator = FullEstimatorInASingleStepWithSmoothMargin(dataset, margin_model_for_estimator, max_stable_model)
    full_estimator.fit()
    full_estimator.margin_function_from_fit.visualize_function(axes=axes, show=False)
plt.show()

# Display all the margin on the same graph for comparison

# Plot the margin functions
# margin_model.margin_function_sample.visualize_2D()
