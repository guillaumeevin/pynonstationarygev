import numpy as np
import properscoring as ps
from cached_property import cached_property

from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_trend.ensemble_simulation.simulation_generator_with_effect.abstract_simulation_with_effect import AbstractSimulationWithEffects
from extreme_trend.one_fold_fit.altitude_group import DefaultAltitudeGroup
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit
from projects.projected_extreme_snowfall.results.combination_utils import \
    load_param_name_to_climate_coordinates_with_effects
from root_utils import get_display_name_from_object_type


class AbstractSimulationFitForEnsemble(object):
    RMSE_METRIC = 'RMSE'
    ABSOLUTE_RELATIVE_DIFFERENCE_METRIC = 'absolute relative difference'
    CRPS_METRIC = 'CRPS'
    WIDTH_METRIC = 'Width of the 90\% uncertainty interval'
    METRICS = [ABSOLUTE_RELATIVE_DIFFERENCE_METRIC, RMSE_METRIC, CRPS_METRIC, WIDTH_METRIC]

    def __init__(self, simulation: AbstractSimulationWithEffects,
                 year_list_to_test,
                 return_period,
                 model_classes,
                 with_effects=True,
                 with_observation=True,
                 color='k'):
        self.color = color
        self.simulation = simulation
        self.return_period = return_period
        self.year_list_to_test = year_list_to_test
        self.x_list_to_test = [self.simulation.get_x_from_year(year) for year in self.year_list_to_test]

        self.with_effects = with_effects
        self.with_observation = with_observation
        assert not ((not with_observation) and with_effects)

        # Default parameters
        self.model_classes = model_classes
        self.fit_method = MarginFitMethod.evgam
        self.altitude_group = DefaultAltitudeGroup(altitudes=[0])
        self.only_models_that_pass_goodness_of_fit_test = False
        self.confidence_interval_based_on_delta_method = False
        self.remove_physically_implausible_models = False
        combination = (2, 2, 0) if self.with_effects else (0, 0, 0)
        self.param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(combination)

    @staticmethod
    def print_one_fold_fit_informations(one_fold_fit):
        print('here info one one fold fit:', one_fold_fit.best_combination, get_display_name_from_object_type(one_fold_fit.best_margin_model))


    def plot_mean_metric(self, ax, metric_name):
        assert metric_name in self.METRICS
        mean_metric = np.mean(self.metric_name_to_all_list[metric_name], axis=0)
        ax.set_xlabel('Years')
        ax.set_xlim((self.year_list_to_test[0], self.year_list_to_test[-1]))
        ax.plot(self.year_list_to_test, mean_metric, label=self.name, color=self.color)
        if self.simulation.nb_simulations < 10:
            print('We do not print uncertainty interval for less than 10 samples')
        else:
            lower_bound, upper_bound = [np.quantile(self.metric_name_to_all_list[metric_name], q, axis=0) for q in [0.25, 0.75]]
            ax.fill_between(self.year_list_to_test, lower_bound, upper_bound, color=self.color, alpha=0.1)

    def load_one_fold_fit(self, dataset, name):
        one_fold_fit = OneFoldFit(massif_name=name, dataset=dataset, models_classes=self.model_classes,
                                  altitude_group=self.altitude_group,
                                  only_models_that_pass_goodness_of_fit_test=self.only_models_that_pass_goodness_of_fit_test,
                                  confidence_interval_based_on_delta_method=self.confidence_interval_based_on_delta_method,
                                  param_name_to_climate_coordinates_with_effects=self.param_name_to_climate_coordinates_with_effects,
                                  fit_method=self.fit_method)
        return one_fold_fit

    @property
    def name(self):
        raise NotImplementedError

    @cached_property
    def metric_name_to_all_list(self):
        all_dict = [ ]
        for i in self.simulation.simulation_ids:
            name = 'Simulation #{} for {}\n'.format(i, get_display_name_from_object_type(type(self)))
            print(self.add_suffix(name))
            all_dict.append(self.compute_metric_name_to_list(i))

        return {metric_name: [d[metric_name] for d in all_dict] for metric_name in self.METRICS}

    def compute_metric_name_to_list(self, simulation_id):
        raise NotImplementedError

    def compute_return_levels(self, margin_function, x):
        gev_params = margin_function.get_params(x)
        return gev_params.return_level(self.return_period)

    def compute_dict(self, margin_functions, margin_functions_uncertainty, true_margin_function):
        crpss_list = []
        rmse_list = []
        absolute_list = []
        width_list = []
        for x in self.x_list_to_test:
            coordinates = np.array([x])
            prediction = np.mean([self.compute_return_levels(f, coordinates) for f in margin_functions])
            absolute_relative_difference, crpss, rmse, width = self.compute_metrics(coordinates, prediction,
                                                                             margin_functions_uncertainty,
                                                                             true_margin_function)

            rmse_list.append(rmse)
            absolute_list.append(absolute_relative_difference)
            crpss_list.append(crpss)
            width_list.append(width)
        return {
            self.RMSE_METRIC: rmse_list,
            self.CRPS_METRIC: crpss_list,
            self.ABSOLUTE_RELATIVE_DIFFERENCE_METRIC: absolute_list,
            self.WIDTH_METRIC: width_list,
        }

    def compute_metrics(self, coordinates, prediction, margin_functions_from_bootstrap, true_margin_function):
        true_value = self.compute_return_levels(true_margin_function, coordinates)
        # Compute rmse
        rmse = np.power(true_value - prediction, 2)
        # Compute absolute relative difference
        absolute_relative_difference = np.abs(100 * (true_value - prediction) / true_value)
        # Compute crpss
        predictions_from_bootstrap = [self.compute_return_levels(f, coordinates) for f in margin_functions_from_bootstrap]
        if len(predictions_from_bootstrap) == 0:
            crpss, width = 0, 0
        else:
            crpss = ps.crps_ensemble(true_value, predictions_from_bootstrap)
            # Compute width metric
            width = np.quantile(predictions_from_bootstrap, 0.95) - np.quantile(predictions_from_bootstrap, 0.05)
        return absolute_relative_difference, crpss, rmse, width

    def add_suffix(self, name):
        if self.with_effects:
            name += 'with effects'
        else:
            name += 'without effects'
        # if self.with_observation:
        #     name += ' with observations'
        # else:
        #     name += ' without observations'
        return name
