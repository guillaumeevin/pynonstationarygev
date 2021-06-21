from random import uniform
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cached_property import cached_property
from matplotlib.lines import Line2D
from numpy.random import beta

from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_rcm_couple_to_color
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.estimator.margin_estimator.utils import fitted_stationary_gev
from extreme_fit.function.margin_function.independent_margin_function import IndependentMarginFunction
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationAndScaleAndShapeTemporalModel
from extreme_fit.model.utils import set_seed_for_test
from projects.projected_extreme_snowfall.results.part_2.average_bias import plot_bias_repartition
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class AbstractSimulationWithEffects(object):

    def __init__(self, nb_simulations,
                 relative_percentage_for_temporal_and_effects=0.1,
                 model_class=NonStationaryLocationAndScaleAndShapeTemporalModel,
                 nb_ensemble_member=20):
        set_seed_for_test()
        self.relative_percentage_for_temporal_and_effects = relative_percentage_for_temporal_and_effects
        self.nb_simulations = nb_simulations
        self.nb_ensemble_member = nb_ensemble_member
        self.model_class = model_class
        self.nb_temporal_steps = 150
        self.coordinates = AbstractCoordinates.from_df(self.load_df_full(with_observations=True))
        self.margin_model = model_class(self.coordinates)
        self.margin_function = self.margin_model.margin_function

        self.simulation_ids = range(self.nb_simulations)
        self.simulation_id_to_margin_function = {}
        self.simulation_id_to_coordinate_to_maxima = {}
        self.simulation_id_to_together_dataset_with_obs = {}
        self.simulation_id_to_separate_datasets_with_obs = {}
        self.simulation_id_to_separate_datasets_without_obs = {}
        self.simulation_id_to_together_dataset_without_obs = {}

        for simulation_id in self.simulation_ids:
            print('simulation', simulation_id)
            acceptable_simulation = False
            while not acceptable_simulation:
                print('looping')
                # Sample once the simulation parameters
                margin_function = self.load_margin_function() # type: IndependentMarginFunction
                # Sample once the observation
                df_full = self.coordinates.df_temporal_coordinates_for_fit(
                    climate_coordinates_with_effects=[AbstractCoordinates.COORDINATE_RCM],
                    drop_duplicates=False)
                df_short = pd.concat([self.coordinates.df_all_coordinates,
                                      self.coordinates.df_coordinate_climate_model.loc[:,
                                      AbstractCoordinates.COORDINATE_RCM]], axis=1)
                couple_coordinate_to_maxima = {}
                for (_, coordinate_full), (_, coordinate_short) in zip(df_full.iterrows(), df_short.iterrows()):
                    couple_coordinate_to_maxima[tuple(coordinate_short)] = margin_function.get_params(
                        coordinate_full).sample(1)

                self.simulation_id_to_margin_function[simulation_id] = margin_function
                self.simulation_id_to_coordinate_to_maxima[simulation_id] = couple_coordinate_to_maxima
                dataset_together_with_obs = self.load_together_dataset(simulation_id, True)
                average_bias = np.array(self.compute_average_bias(dataset_together_with_obs)[-1])
                average_bias_reference = np.array(self.average_bias_reference)
                acceptable_simulation = np.linalg.norm(average_bias - average_bias_reference) < 5

            # Load the dict
            self.simulation_id_to_together_dataset_with_obs[simulation_id] = self.load_together_dataset(simulation_id, True)
            self.simulation_id_to_separate_datasets_with_obs[simulation_id] = self.load_separate_datasets(simulation_id, True)
            self.simulation_id_to_separate_datasets_without_obs[simulation_id] = self.load_separate_datasets(simulation_id, False)
            self.simulation_id_to_together_dataset_without_obs[simulation_id] = self.load_together_dataset(simulation_id, False)

    def load_separate_datasets(self, simulation_id, with_observations):
        datasets = []
        for j in range(self.nb_ensemble_member):
            coordinates = AbstractCoordinates.from_df(self.load_df_separate(j, with_observations))
            datasets.append(self.load_dataset(coordinates, simulation_id))
        return datasets

    def load_together_dataset(self, simulation_id, with_observations):
        coordinates = AbstractCoordinates.from_df(self.load_df_full(with_observations))
        return self.load_dataset(coordinates, simulation_id)

    def load_dataset(self, coordinates, simulation_id):
        df_short = pd.concat([coordinates.df_all_coordinates,
                              coordinates.df_coordinate_climate_model.loc[:, AbstractCoordinates.COORDINATE_RCM]],
                             axis=1)

        coordinate_to_maxima = self.simulation_id_to_coordinate_to_maxima[simulation_id]
        maxima_list = [coordinate_to_maxima[tuple(coordinate_short)] for _, coordinate_short in df_short.iterrows()]
        df_maxima_gev = pd.DataFrame.from_dict({'obs_gev': maxima_list})
        df_maxima_gev.index = coordinates.index
        observations = AbstractSpatioTemporalObservations(df_maxima_gev=df_maxima_gev)
        return AbstractDataset(observations, coordinates)

    def load_df_separate(self, j, with_observations=True):
        df_final = self.load_df_observations(with_observations)
        df = self.load_df_ensemble_member(j)
        df_final = df_final.append(df, ignore_index=True)
        if with_observations:
            assert len(df_final) == 150 + 61
        else:
            assert len(df_final) == 150
        df_final.index = np.arange(0, len(df_final))
        return df_final

    def load_df_full(self, with_observations=True):
        df_final = self.load_df_observations(with_observations)
        for j in range(self.nb_ensemble_member):
            df = self.load_df_ensemble_member(j)
            df_final = df_final.append(df, ignore_index=True)
        if with_observations:
            assert len(df_final) == 150 * self.nb_ensemble_member + 61
        else:
            assert len(df_final) == 150 * self.nb_ensemble_member
        df_final.index = np.arange(0, len(df_final))
        assert len(df_final.columns) == 2
        return df_final

    def load_df_ensemble_member(self, j):
        df = self.load_df_basic()
        name = self.ensemble_member_idx_to_name(j)
        df[AbstractCoordinates.COORDINATE_RCM] = name
        return df

    def load_df_observations(self, with_observations):
        if with_observations:
            df = self.load_df_basic()
            # For the observations we keep only the data between 1959 and 2019
            df = df.iloc[8:-81]
            assert len(df) == 61
            return df
        else:
            return pd.DataFrame()

    def load_df_basic(self):
        df = ConsecutiveTemporalCoordinates.from_nb_temporal_steps(
            nb_temporal_steps=self.nb_temporal_steps, start=0, end=1).df_all_coordinates
        df[AbstractCoordinates.COORDINATE_RCM] = None
        return df

    def ensemble_member_idx_to_name(self, j):
        return 'RCM_{}'.format(j + 1)

    def load_margin_function(self) -> IndependentMarginFunction:
        raise NotImplementedError

    def plot_simulation_parameter(self, gev_param_name, simulation_ids, plot_ensemble_members=False):
        ax = plt.gca()
        x_list = np.linspace(0, 1, num=150)
        # colors = list(gcm_rcm_couple_to_color.values())
        # colors = ['lightpink', 'violet', 'm', 'darkmagenta']
        colors = ['red', 'blue', 'cyan', 'y', 'k', 'green']
        colors += ['red', 'blue', 'cyan', 'y', 'k', 'green']
        colors += ['lightpink', 'violet', 'm', 'darkmagenta']
        assert len(simulation_ids) <= len(colors)
        for color, simulation_id in zip(colors, simulation_ids):
            margin_function = self.simulation_id_to_margin_function[simulation_id]

            # Plot observations
            x_list_past_observation = [x for x in x_list if self.year_from_x(x) < 2020]
            x_list_projected_observation = [x for x in x_list if self.year_from_x(x) >= 2020]
            y_list = [self.get_params_simulation(margin_function, x, None, gev_param_name) for x in
                      x_list_past_observation]
            ax.plot(x_list_past_observation, y_list, color=color, linewidth=4,
                    label='Simulation #{}'.format(simulation_id + 1))
            y_list = [self.get_params_simulation(margin_function, x, None, gev_param_name) for x in
                      x_list_projected_observation]
            ax.plot(x_list_projected_observation, y_list, color=color, linewidth=4, linestyle='dashed')

            # Plot ensemble members
            if plot_ensemble_members:
                for j in range(self.nb_ensemble_member):
                    y_list = [self.get_params_simulation(margin_function, x, j, gev_param_name) for x in x_list]
                    ax.plot(x_list, y_list, color=color, linewidth=1, linestyle='dotted')

        self.set_fake_x_axis(ax)
        ax2 = ax.twinx()
        legend_elements = [
            Line2D([0], [0], color='k', lw=4, label="Past observation", linestyle='-'),
            Line2D([0], [0], color='k', lw=4, label="Projected observation", linestyle='dashed'),
            Line2D([0], [0], color='k', lw=1, label="Ensemble member", linestyle='dotted'),
        ]
        if gev_param_name is GevParams.SHAPE:
            legend_elements = legend_elements[:1]
        size = 7
        ax2.legend(handles=legend_elements, loc='upper right', prop={'size': size}, handlelength=5)
        ax2.set_yticks([])

        ylabel = '100-year return level' if gev_param_name is None \
            else '{} parameter'.format(GevParams.full_name_from_param_name(gev_param_name))
        ax.set_ylabel(ylabel)
        self.visualizer.plot_name = '{} simulation'.format(ylabel)
        ax.legend(loc='upper left', prop={'size': size})
        self.visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)
        plt.close()

    def set_fake_x_axis(self, ax):
        ax.set_xlabel('Years')
        plt.tick_params(axis='both', which='major', labelsize=7)
        n = self.nb_temporal_steps - 1
        ticks = [-1 / n] + [(9 + 10 * i) / n for i in range(15)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([1950 + 10 * i for i in range(16)])
        ax.set_xlim((ticks[0], ticks[-1]))

    def get_maxima_between_1959_and_2019(self, dataset, j):
        maxima_list, x_list = self.get_maxima_and_x_list(dataset, j)
        maxima_between_1959_and_2019 = [m for m, x in zip(maxima_list, x_list) if 1959 <= self.year_from_x(x) <= 2019]
        assert len(maxima_between_1959_and_2019) == 61, len(maxima_between_1959_and_2019)
        return maxima_between_1959_and_2019

    def get_params_simulation(self, margin_function, x, j, gev_param_name):
        if j is None:
            coordinate = np.array([x])
        else:
            coordinate = np.zeros(self.nb_ensemble_member + 1)
            coordinate[0] = x
            coordinate[j + 1] = 1
        gev_params = margin_function.get_params(coordinate)  # type: GevParams
        if gev_param_name in GevParams.PARAM_NAMES:
            param = gev_params.to_dict()[gev_param_name]
        else:
            param = gev_params.return_level(return_period=100)
        return param

    def get_x_from_year(self, year):
        return (year - 1951) / self.nb_temporal_steps

    def year_from_x(self, x):
        return int(x * (self.nb_temporal_steps - 1) + 1951)

    def get_index_from_year_and_j(self, year, j):
        if j is None:
            return year - 1951
        else:
            return year - 1951 + (j + 1) * 150

    @staticmethod
    def get_list(dataset, ind):
        x_list = [e[0] for e in dataset.coordinates.df_all_coordinates.loc[ind].values]
        maxima_list = [e[0] for e in dataset.observations.df_maxima_gev.loc[ind].values[:, 0]]
        return maxima_list, x_list

    def plot_time_series(self, simulation_id):

        ax = plt.gca()
        dataset = self.simulation_id_to_together_dataset_with_obs[simulation_id]
        # Plot the ensemble member
        colors = list(gcm_rcm_couple_to_color.values())[::-1]
        for j, color in zip(list(range(self.nb_ensemble_member)), colors):
            maxima_list, x_list = self.get_maxima_and_x_for_ensemble_member_j(dataset, j)
            fake_index = self.nb_ensemble_member - j
            ax.plot(x_list, maxima_list, color=color, linewidth=2, label='Ensemble member #{}'.format(fake_index))
        # Plot the observation on top
        maxima_list, x_list = self.get_maxima_and_x_list_for_observations(dataset)
        ax.plot(x_list, maxima_list, color='black', linewidth=4, label='Observation')
        self.set_fake_x_axis(ax)
        ax.set_ylabel('Simulated annual maxima for simulation #{}'.format(simulation_id + 1))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], prop={'size': 6}, ncol=3)
        self.visualizer.plot_name = 'observations from simulation {}'.format(simulation_id + 1)
        self.visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)
        plt.close()

    def plot_bias(self, simulation_id):
        ax = plt.gca()
        dataset = self.simulation_id_to_together_dataset_with_obs[simulation_id]
        all_biases, average_bias = self.compute_average_bias(dataset)

        colors = list(gcm_rcm_couple_to_color.values())[::-1]
        for j, (biases, color) in enumerate(zip(all_biases, colors)):
            xi, yi = biases
            fake_index = self.nb_ensemble_member - j
            name = 'Ensemble member #{}'.format(fake_index)
            ax.scatter([xi], [yi], color=color, marker='o', label=name)

        plot_bias_repartition(average_bias, ax, 'observation', skip_percent=False)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], prop={'size': 7}, loc='lower right', ncol=1)

        self.visualizer.plot_name = 'bias from simulation {}'.format(simulation_id + 1)
        self.visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

        plt.close()

    def compute_average_bias(self, dataset):
        annual_maxima1 = self.get_maxima_between_1959_and_2019(dataset, None)
        all_biases = []
        for j in range(self.nb_ensemble_member):
            annual_maxima2 = self.get_maxima_between_1959_and_2019(dataset, j)
            biases = []
            for f in [np.mean, np.std]:
                moment_ref, moment_comparison = [f(maxima) for maxima in [annual_maxima1, annual_maxima2]]
                bias = moment_comparison - moment_ref
                bias *= 100 / moment_ref
                biases.append(bias)
            all_biases.append(biases)
        all_biases = np.array(all_biases)
        average_bias = np.mean(all_biases, axis=0)
        return all_biases, average_bias

    def get_maxima_and_x_list(self, dataset, j):
        if j is None:
            return self.get_maxima_and_x_list_for_observations(dataset)
        else:
            return self.get_maxima_and_x_for_ensemble_member_j(dataset, j)

    def get_maxima_and_x_list_for_observations(self, dataset):
        ind = self.coordinates.df_coordinate_climate_model.isnull().any(axis=1)
        maxima_list, x_list = self.get_list(dataset, ind)
        return maxima_list, x_list

    def get_maxima_and_x_for_ensemble_member_j(self, dataset, j):
        ind = self.coordinates.df_coordinate_climate_model[
                  AbstractCoordinates.COORDINATE_RCM] == self.ensemble_member_idx_to_name(j)
        maxima_list, x_list = self.get_list(dataset, ind)
        return maxima_list, x_list

    @property
    def summary_parameter(self):
        raise NotImplementedError

    @cached_property
    def visualizer(self):
        return StudyVisualizer(SafranSnowfall1Day(), show=False, save_to_file=True)

    def sample_around(self, value):
        assert value != 0
        # uniform sampling around the value
        bound = np.abs(value) * self.relative_percentage_for_temporal_and_effects
        new_value = uniform(a=-bound, b=bound)
        return value + new_value
        # normal sampling could be done..

    def sample_uniform(self, bound):
        return uniform(a=-bound, b=bound)

    def _sample_uniform(self, bound_left, bound_right):
        return uniform(bound_left, bound_right)

    def sample_uniform_scale(self, alpha):
        return self._sample_uniform(np.log(1 - alpha), np.log(1 + alpha))
