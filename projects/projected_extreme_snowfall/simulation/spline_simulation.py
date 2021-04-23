from extreme_fit.distribution.gev.gev_params import GevParams
import numpy as np
from extreme_fit.estimator.margin_estimator.utils import fitted_linear_margin_estimator
from extreme_fit.function.param_function.param_function import SplineParamFunction
from extreme_fit.function.param_function.spline_coef import SplineAllCoef, SplineCoef
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearLocationModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
import matplotlib.pyplot as plt

from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations
import pandas as pd

nb_temporal_steps = 20


def generate_df_coordinates():
    return ConsecutiveTemporalCoordinates.from_nb_temporal_steps(nb_temporal_steps=nb_temporal_steps)


coordinates = generate_df_coordinates()


def get_margin_function_from_fit(model_class, gev_parameter):
    ground_truth_model = load_ground_truth_model(gev_parameter, model_class)
    df = pd.DataFrame()
    # nb_samples should remain equal to 1
    nb_samples = 1
    for _ in range(nb_samples):
        maxima = [ground_truth_model.margin_function.get_params(c).sample(1)[0]
                  for c in coordinates.df_all_coordinates.values]
        df2 = pd.DataFrame(data=maxima, index=coordinates.index)
        df = pd.concat([df, df2], axis=0)
    index = pd.RangeIndex(0, nb_samples * nb_temporal_steps)
    df.index = index
    observations = AbstractSpatioTemporalObservations(df_maxima_gev=df)

    df = pd.concat([generate_df_coordinates().df_all_coordinates for _ in range(nb_samples)], axis=0)
    df.index= index
    dataset = AbstractDataset(observations=observations, coordinates=AbstractTemporalCoordinates.from_df(df))
    estimator = fitted_linear_margin_estimator(model_class,
                                               coordinates, dataset,
                                               starting_year=None,
                                               fit_method=MarginFitMethod.evgam)
    return estimator.margin_function_from_fit


def get_params_from_margin(margin_function, gev_parameter, x):
    return [margin_function.get_params(np.array([e])).to_dict()[gev_parameter]
            for e in x]


def plot_model_against_estimated_models(model_class, gev_parameter):
    x = coordinates.t_coordinates
    x = np.linspace(x[0], x[-1], num=100)

    ground_truth_model = load_ground_truth_model(gev_parameter, model_class)
    ground_truth_params = get_params_from_margin(ground_truth_model.margin_function, gev_parameter, x)
    plt.plot(x, ground_truth_params)

    for _ in range(10):
        params = get_params_from_margin(get_margin_function_from_fit(model_class, gev_parameter), gev_parameter, x)
        plt.plot(x, params)

    plt.grid()
    plt.show()


def load_ground_truth_model(gev_parameter, model_class):
    # knots = [-75.3, -0.15, 75, 150.15, 225.3]
    # knots = [-4.96980e+01, -9.90000e-02,  4.95000e+01,  9.90990e+01,  1.48698e+02]
    shift = 10
    knots = [-shift, 0, nb_temporal_steps // 2, nb_temporal_steps, nb_temporal_steps + shift]
    spline_coef = SplineCoef(param_name=gev_parameter, knots=knots, coefficients=[10, 30, -50])
    spline_all_coef = SplineAllCoef(gev_parameter, {0: spline_coef})
    spline_param_function = SplineParamFunction(dim_and_degree=[(0, 1)], coef=spline_all_coef)
    model = model_class(coordinates=coordinates)
    model.margin_function.param_name_to_param_function[gev_parameter] = spline_param_function
    return model


if __name__ == '__main__':
    plot_model_against_estimated_models(NonStationaryTwoLinearLocationModel, GevParams.LOC)
