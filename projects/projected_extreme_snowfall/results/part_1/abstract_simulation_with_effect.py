from random import uniform

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
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationAndScaleAndShapeTemporalModel
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
        self.relative_percentage_for_temporal_and_effects = relative_percentage_for_temporal_and_effects
        self.nb_simulations = nb_simulations
        self.nb_ensemble_member = nb_ensemble_member
        self.model_class = model_class
        self.nb_temporal_steps = 150
        self.coordinates = AbstractCoordinates.from_df(self.load_df())
        self.margin_model = model_class(self.coordinates)
        self.margin_function = self.margin_model.margin_function

        print()

        # Sample the simulation parameters
        self.simulation_id_to_margin_function = {simulation_id: self.load_margin_function()
                                                 for simulation_id in range(self.nb_simulations)}

        # Sample once the observation from the simulation parameters
        self.simulation_id_to_dataset = {}
        for simulation_id, margin_function in self.simulation_id_to_margin_function.items():
            df = self.coordinates.df_temporal_coordinates_for_fit(climate_coordinates_with_effects=[AbstractCoordinates.COORDINATE_RCM],
                                                               drop_duplicates=False)
            maxima_list = [margin_function.get_params(coordinate).sample(1) for _, coordinate in df.iterrows()]
            df_maxima_gev = pd.DataFrame.from_dict({'obs_gev': maxima_list})
            df_maxima_gev.index = self.coordinates.index
            observations = AbstractSpatioTemporalObservations(df_maxima_gev=df_maxima_gev)
            dataset = AbstractDataset(observations, self.coordinates)
            self.simulation_id_to_dataset[simulation_id] = dataset

    def load_df(self):
        df = ConsecutiveTemporalCoordinates.from_nb_temporal_steps(
            nb_temporal_steps=self.nb_temporal_steps, start=0, end=1).df_all_coordinates
        df[AbstractCoordinates.COORDINATE_RCM] = None
        # For the observations we keep only the data between 1959 and 2019
        df_final = df.copy().iloc[8:-81]
        assert len(df_final) == 61
        for j in range(self.nb_ensemble_member):
            name = self.ensemble_member_idx_to_name(j)
            df[AbstractCoordinates.COORDINATE_RCM] = name
            df_final = df_final.append(df.copy(), ignore_index=True)
        # df_final = pd.concat(df_list, axis=0, ignore_index=True, join='inner')
        df_final.index = np.arange(1, len(df_final)+1)
        assert len(df_final.columns) == 2
        return df_final

    def ensemble_member_idx_to_name(self, j):
        return 'RCM_{}'.format(j+1)

    def load_margin_function(self):
        # Sample the non-stationary parameters
        coef_dict = dict()
        coef_dict['locCoeff1'] = 10
        coef_dict['scaleCoeff1'] = 1
        shape = beta(6, 9) - 0.5
        coef_dict['shapeCoeff1'] = shape
        coef_dict['tempCoeffLoc1'] = self.sample_around(coef_dict['locCoeff1'])
        coef_dict['tempCoeffScale1'] = self.sample_around(coef_dict['scaleCoeff1'])
        coef_dict['tempCoeffShape1'] = self.sample_around(coef_dict['shapeCoeff1'])
        # Climatic effects
        param_name_to_climate_coordinates_with_effects = {
            GevParams.LOC: [AbstractCoordinates.COORDINATE_RCM],
            GevParams.SCALE: [AbstractCoordinates.COORDINATE_RCM],
            GevParams.SHAPE: None,
        }
        param_name_to_ordered_climate_effects = {
            GevParams.LOC: [self.sample_around(coef_dict['locCoeff1']) for _ in range(self.nb_ensemble_member)],
            GevParams.SCALE: [self.sample_around(coef_dict['scaleCoeff1']) for _ in range(self.nb_ensemble_member)],
            GevParams.SHAPE: [],
        }
        # Load margin function
        margin_function = type(self.margin_function).from_coef_dict(self.coordinates,
                                                                    self.margin_function.param_name_to_dims,
                                                                    coef_dict,
                                                                    param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects,
                                                                    param_name_to_ordered_climate_effects=param_name_to_ordered_climate_effects)
        return margin_function

    def plot_simulation_parameter(self, gev_param_name, simulation_ids, plot_ensemble_members=False):
        ax = plt.gca()
        x_list = np.linspace(0, 1, num=50)
        colors = ['r', 'g', 'b', 'orange']
        assert len(simulation_ids) <= len(colors)
        for color, simulation_id in zip(colors, simulation_ids):
            margin_function = self.simulation_id_to_margin_function[simulation_id]

            # Plot observations
            y_list = [self.get_params_simulation(margin_function, x, None, gev_param_name) for x in x_list]
            ax.plot(x_list, y_list, color=color, linewidth=4, label='Simulation #{}'.format(simulation_id+1))

            # Plot ensemble members
            if plot_ensemble_members:
                for j in range(self.nb_ensemble_member):
                    y_list = [self.get_params_simulation(margin_function, x, j, gev_param_name) for x in x_list]
                    ax.plot(x_list, y_list, color=color, linewidth=1, linestyle='dotted')

        self.set_fake_x_axis(ax)
        ax2 = ax.twinx()
        legend_elements = [
            Line2D([0], [0], color='k', lw=4, label="Observation", linestyle='-'),
            Line2D([0], [0], color='k', lw=1, label="Ensemble member", linestyle='dotted'),
        ]
        if gev_param_name is GevParams.SHAPE:
            legend_elements = legend_elements[:1]
        ax2.legend(handles=legend_elements, loc='upper right', prop={'size': 7})
        ax2.set_yticks([])

        ylabel = '100-year return level' if gev_param_name is None \
            else '{} parameter'.format(GevParams.full_name_from_param_name(gev_param_name))
        ax.set_ylabel(ylabel)
        self.visualizer.plot_name = '{} simulation'.format(ylabel)
        ax.legend(loc='upper left', prop={'size': 7})
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
        return x * self.nb_temporal_steps + 1951

    def get_index_from_year_and_j(self, year, j):
        if j is None:
            return year - 1951
        else:
            return year - 1951 + (j+1) * 150

    def plot_time_series(self, simulation_id):
        def get_list(dataset, ind):
            x_list = [e[0] for e in dataset.coordinates.df_all_coordinates.loc[ind].values]
            maxima_list = [e[0] for e in dataset.observations.df_maxima_gev.loc[ind].values[:, 0]]
            return maxima_list, x_list

        ax = plt.gca()
        dataset = self.simulation_id_to_dataset[simulation_id]
        # Plot the ensemble member
        colors = list(gcm_rcm_couple_to_color.values())
        for j, color in zip(list(range(self.nb_ensemble_member)), colors):
            ind = self.coordinates.df_coordinate_climate_model[AbstractCoordinates.COORDINATE_RCM] == self.ensemble_member_idx_to_name(j)
            maxima_list, x_list = get_list(dataset, ind)
            ax.plot(x_list, maxima_list, color=color, linewidth=2, label='Ensemble member #{}'.format(j+1))
        # Plot the observation on top
        ind = self.coordinates.df_coordinate_climate_model.isnull().any(axis=1)
        maxima_list, x_list = get_list(dataset, ind)
        ax.plot(x_list, maxima_list, color='black', linewidth=4, label='Observation')
        self.set_fake_x_axis(ax)
        ax.set_ylabel('Simulated annual maxima')
        ax.legend(prop={'size': 6}, ncol=3)
        self.visualizer.plot_name = 'observations from simulation {}'.format(simulation_id+1)
        self.visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)
        plt.close()

    @cached_property
    def visualizer(self):
        return StudyVisualizer(SafranSnowfall1Day(), show=False, save_to_file=True)


    def sample_around(self, value):
        assert value != 0
        # uniform sampling around the value
        bound = np.abs(value) * self.relative_percentage_for_temporal_and_effects
        new_value = uniform(a=-bound, b=bound)
        return new_value
        # normal sampling could be done..
