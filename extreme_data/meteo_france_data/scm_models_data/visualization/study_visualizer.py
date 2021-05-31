import os
import matplotlib.colors as mcolors
import os.path as op
from collections import OrderedDict
from multiprocessing.pool import Pool
from random import sample, seed
from typing import Dict, Tuple

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from extreme_data.meteo_france_data.adamont_data.abstract_adamont_study import AbstractAdamontStudy
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_str
from extreme_data.meteo_france_data.scm_models_data.visualization.plot_utils import load_plot
from extreme_fit.estimator.margin_estimator.utils import fitted_stationary_gev
from extreme_fit.model.margin_model.utils import fitmethod_to_str
from extreme_fit.model.result_from_model_fit.result_from_extremes.eurocode_return_level_uncertainties import \
    EurocodeConfidenceIntervalFromExtremes, compute_eurocode_confidence_interval
from extreme_data.meteo_france_data.scm_models_data.abstract_extended_study import AbstractExtendedStudy
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.utils import create_adjusted_axes
from extreme_fit.distribution.abstract_params import AbstractParams
from extreme_fit.estimator.full_estimator.abstract_full_estimator import \
    FullEstimatorInASingleStepWithSmoothMargin
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import \
    LinearNonStationaryLocationMarginModel, \
    LinearStationaryMarginModel
from extreme_fit.function.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from extreme_fit.function.param_function.param_function import AbstractParamFunction
from extreme_fit.model.max_stable_model.abstract_max_stable_model import CovarianceFunction
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.distribution.gpd.gpd_params import GpdParams
from extreme_fit.distribution.gpd.gpdmle_fit import GpdMleFit
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.abstract_spatio_temporal_coordinates import \
    AbstractSpatioTemporalCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import AnnualMaxima
from test.test_utils import load_test_max_stable_models
from root_utils import get_display_name_from_object_type, VERSION_TIME, float_to_str_with_only_some_significant_digits, \
    cached_property, NB_CORES

BLOCK_MAXIMA_DISPLAY_NAME = 'block maxima '


class VisualizationParameters(object):

    def __init__(self, save_to_file=False, only_one_graph=False, only_first_row=False, show=True):
        self.only_first_row = only_first_row
        self.only_one_graph = only_one_graph
        self.save_to_file = save_to_file

        # PLOT ARGUMENTS
        self.show = False if self.save_to_file else show
        if self.only_one_graph:
            self.figsize = (6.0, 4.0)
        elif self.only_first_row:
            self.figsize = (8.0, 6.0)
        else:
            self.figsize = (16.0, 10.0)
        self.subplot_space = 0.5
        self.coef_zoom_map = 1


class StudyVisualizer(VisualizationParameters):

    def __init__(self, study: AbstractStudy, show=True, save_to_file=False, only_one_graph=False, only_first_row=False,
                 vertical_kde_plot=False, year_for_kde_plot=None, plot_block_maxima_quantiles=False,
                 temporal_non_stationarity=False, transformation_class=None, verbose=False, multiprocessing=False,
                 complete_non_stationary_trend_analysis=False):
        super().__init__(save_to_file, only_one_graph, only_first_row, show)
        self.nb_cores = 7
        self.massif_id_to_smooth_maxima = {}
        self.temporal_non_stationarity = temporal_non_stationarity
        self.only_first_row = only_first_row
        self.only_one_graph = only_one_graph
        self.save_to_file = save_to_file
        self.study = study
        self.plot_name = None

        self.multiprocessing = multiprocessing
        self.verbose = verbose
        self.complete_non_stationary_trend_analysis = complete_non_stationary_trend_analysis

        # Load some attributes
        self._dataset = None
        self._coordinates = None
        self._observations = None

        self.default_covariance_function = CovarianceFunction.powexp
        self.transformation_class = transformation_class

        # KDE PLOT ARGUMENTS
        self.vertical_kde_plot = vertical_kde_plot
        self.year_for_kde_plot = year_for_kde_plot
        self.plot_block_maxima_quantiles = plot_block_maxima_quantiles

        self.window_size_for_smoothing = 1  # other value could be
        self.number_of_top_values = 10  # 1 if we just want the maxima

        # Modify some class attributes
        # Remove some assert
        AbstractParamFunction.OUT_OF_BOUNDS_ASSERT = False
        # INCREASE THE TEMPORAL STEPS FOR VISUALIZATION
        AbstractMarginFunction.VISUALIZATION_TEMPORAL_STEPS = 5

        # Change point parameters
        self.trend_test_class_for_change_point_test = None
        self.starting_years_for_change_point_test = None
        self.nb_massif_for_change_point_test = None

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = AbstractDataset(self.observations, self.coordinates)
        return self._dataset

    @property
    def spatial_coordinates(self):
        return AbstractSpatialCoordinates.from_df(df=self.study.df_massifs_longitude_and_latitude,
                                                  transformation_class=self.transformation_class)

    @property
    def temporal_coordinates(self):
        start, stop = self.study.start_year_and_stop_year
        nb_steps = stop - start + 1
        temporal_coordinates = ConsecutiveTemporalCoordinates.from_nb_temporal_steps(nb_temporal_steps=nb_steps,
                                                                                     start=start,
                                                                                     transformation_class=self.transformation_class)
        return temporal_coordinates

    @property
    def spatio_temporal_coordinates(self):
        return AbstractSpatioTemporalCoordinates.from_spatial_coordinates_and_temporal_coordinates(
            spatial_coordinates=self.spatial_coordinates, temporal_coordinates=self.temporal_coordinates)

    @property
    def coordinates(self):
        if self._coordinates is None:
            if self.temporal_non_stationarity:
                # Build spatio temporal coordinates from a spatial coordinates and a temporal coordinates
                coordinates = self.spatio_temporal_coordinates
            else:
                # By default otherwise, we only keep the spatial coordinates
                coordinates = self.spatial_coordinates
            self._coordinates = coordinates
        return self._coordinates

    @property
    def observations(self):
        if self._observations is None:
            self._observations = self.study.observations_annual_maxima
            if self.temporal_non_stationarity:
                self._observations.convert_to_spatio_temporal_index(self.coordinates)
                if self.verbose:
                    self._observations.print_summary()
        return self._observations

    def observation_massif_id(self, massif_id):
        annual_maxima = [data[massif_id] for data in self.study.year_to_annual_maxima.values()]
        df_annual_maxima = pd.DataFrame(annual_maxima, index=self.temporal_coordinates.index)
        observation_massif_id = AnnualMaxima(df_maxima_gev=df_annual_maxima)
        if self.verbose:
            observation_massif_id.print_summary()
        return observation_massif_id

    # PLOT FOR SEVERAL MASSIFS

    def visualize_massif_graphs(self, visualize_function, specified_massif_ids=None):
        if self.only_one_graph:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            visualize_function(ax, 0)
        else:
            nb_columns = 5
            nb_plots = len(self.study.study_massif_names) if specified_massif_ids is None else len(specified_massif_ids)
            nb_rows = 1 if self.only_first_row else math.ceil(nb_plots / nb_columns)
            fig, axes = plt.subplots(nb_rows, nb_columns, figsize=self.figsize)
            fig.subplots_adjust(hspace=self.subplot_space, wspace=self.subplot_space)
            if self.only_first_row:
                for massif_id, massif_name in enumerate(self.study.study_massif_names[:nb_columns]):
                    ax = axes[massif_id]
                    visualize_function(ax, massif_id)
            else:
                if specified_massif_ids is None:
                    specified_massif_ids = list(range(len(self.study.study_massif_names)))
                for j, massif_id in enumerate(specified_massif_ids):
                    row_id, column_id = j // nb_columns, j % nb_columns
                    if len(specified_massif_ids) < nb_columns:
                        ax = axes[column_id]
                    else:
                        ax = axes[row_id, column_id]
                    visualize_function(ax, massif_id)

    # EXPERIMENTAL LAW

    def visualize_all_experimental_law(self):
        self.visualize_massif_graphs(self.visualize_experimental_law)
        self.plot_name = ' Empirical distribution \n'
        self.plot_name += 'with data from the 23 mountain chains of the French Alps ' if self.year_for_kde_plot is None else \
            'for the year {}'.format(self.year_for_kde_plot)
        self.show_or_save_to_file()

    def visualize_experimental_law(self, ax, massif_id):
        # Display the experimental law for a given massif
        all_massif_data = self.get_all_massif_data(massif_id)

        # Display an histogram on the background (with 100 bins, for visibility, and to check 0.9 quantiles)
        ax2 = ax.twiny() if self.vertical_kde_plot else ax.twinx()
        color_hist = 'k'
        orientation = "horizontal" if self.vertical_kde_plot else 'vertical'
        weights = np.ones_like(all_massif_data) / float(len(all_massif_data))
        ax2.hist(all_massif_data, weights=weights, bins=50,
                 histtype='step', color=color_hist, orientation=orientation)
        label_function = ax2.set_xlabel if self.vertical_kde_plot else ax2.set_ylabel
        # Do not display this label in the vertical plot
        if not self.vertical_kde_plot:
            label_function('normalized histogram', color=color_hist)

        # Kde plot, and retrieve the data forming the line
        color_kde = 'b'
        sns.kdeplot(all_massif_data, ax=ax, color=color_kde, vertical=self.vertical_kde_plot)
        ax.set(ylim=0)
        ax.set(xlim=0)
        data_x, data_y = ax.lines[0].get_data()

        # Plot the mean and median points
        name_to_xlevel_and_color = OrderedDict()
        name_to_xlevel_and_color['median'] = (np.median(all_massif_data), 'chartreuse')
        name_to_xlevel_and_color['mean'] = (np.mean(all_massif_data), 'g')
        # Plot some specific "extreme" quantiles with their color
        for p, color, name in zip(AbstractParams.QUANTILE_P_VALUES, AbstractParams.QUANTILE_COLORS,
                                  AbstractParams.QUANTILE_NAMES):
            x_level = all_massif_data[int(p * len(all_massif_data))]
            name_to_xlevel_and_color[name] = (x_level, color)
            # Plot some additional quantiles from the correspond Annual Maxima law
            if self.plot_block_maxima_quantiles:
                # This formula can only be applied if we have a daily time serie
                assert len(self.study.year_to_daily_time_serie_array[1958]) in [365, 366]
                p = p ** (1 / 365)
                x_level = all_massif_data[int(p * len(all_massif_data))]
                name_to_xlevel_and_color[BLOCK_MAXIMA_DISPLAY_NAME + name] = (x_level, color)
        # Plot the maxima
        name_to_xlevel_and_color['maxima'] = (np.max(all_massif_data), 'darkmagenta')

        for name, (xi, color) in name_to_xlevel_and_color.items():
            if self.vertical_kde_plot:
                yi = xi
                xi = np.interp(yi, data_y, data_x)
            else:
                yi = np.interp(xi, data_x, data_y)
            marker = "x" if BLOCK_MAXIMA_DISPLAY_NAME in name else "o"
            ax.scatter([xi], [yi], color=color, marker=marker, label=name)

        label_function = ax.set_xlabel if self.vertical_kde_plot else ax.set_ylabel
        label_function('Probability Density function f(x)', color=color_kde)

        xlabel = 'x = {}'.format(self.study.title) if self.only_one_graph else 'x'
        label_function = ax.set_ylabel if self.vertical_kde_plot else ax.set_xlabel
        label_function(xlabel)

        # Take all the ticks
        # sorted_x_levels = sorted(list([x_level for x_level, _ in name_to_xlevel_and_color.values()]))
        # extraticks = [float(float_to_str_with_only_some_significant_digits(x, nb_digits=2))
        #               for x in sorted_x_levels]
        # Display only some specific ticks
        extraticks_names = ['mean', AbstractParams.QUANTILE_10, AbstractParams.QUANTILE_100, 'maxima']
        if self.plot_block_maxima_quantiles:
            extraticks_names += [name for name in name_to_xlevel_and_color.keys() if BLOCK_MAXIMA_DISPLAY_NAME in name]
        extraticks = [name_to_xlevel_and_color[name][0] for name in extraticks_names]

        set_ticks_function = ax.set_yticks if self.vertical_kde_plot else ax.set_xticks
        # Round up the ticks with a given number of significative digits
        extraticks = [float(float_to_str_with_only_some_significant_digits(t, nb_digits=2)) for t in extraticks]
        set_ticks_function(extraticks)
        if not self.only_one_graph:
            ax.set_title(self.study.study_massif_names[massif_id])
        ax.legend()

    def get_all_massif_data(self, massif_id):
        if self.year_for_kde_plot is not None:
            all_massif_data = self.study.year_to_daily_time_serie_array[self.year_for_kde_plot][:, massif_id]
        else:
            all_massif_data = np.concatenate(
                [data[:, massif_id] for data in self.study.year_to_daily_time_serie_array.values()])
        all_massif_data = np.sort(all_massif_data)
        return all_massif_data

    @property
    def starting_years(self):
        # Starting years are any potential between the start_year and the end_year.
        # start_year is a potential starting_year
        # end_year is not a potential starting_year
        start_year, stop_year = self.study.start_year_and_stop_year
        return list(range(start_year, stop_year))

    def massif_name_to_altitude_and_eurocode_level_uncertainty(self, model_class, massif_names,
                                                               ci_method, effective_temporal_covariate) -> Dict[
        str, Tuple[int, EurocodeConfidenceIntervalFromExtremes]]:
        massif_ids_and_names = [(massif_id, massif_name) for massif_id, massif_name in
                                enumerate(self.study.study_massif_names) if massif_name in massif_names]
        arguments = [
            [self.smooth_maxima_x_y(massif_id), model_class, ci_method, effective_temporal_covariate] for
            massif_id, _ in massif_ids_and_names]
        if self.multiprocessing:
            with Pool(NB_CORES) as p:
                res = p.starmap(compute_eurocode_confidence_interval, arguments)
        else:
            res = [compute_eurocode_confidence_interval(*argument) for argument in arguments]
        altitudes_and_res = [(self.study.altitude, r) for r in res]
        massif_name_to_eurocode_return_level_uncertainty = OrderedDict(
            zip([massif_name for _, massif_name in massif_ids_and_names], altitudes_and_res))
        return massif_name_to_eurocode_return_level_uncertainty

    def visualize_all_mean_and_max_graphs(self):
        specified_massif_ids = [self.study.study_massif_names.index(massif_name)
                                for massif_name in
                                sorted(self.study.study_massif_names, key=lambda s: self.massif_name_to_first_score[s])]
        self.visualize_massif_graphs(self.visualize_mean_and_max_graph,
                                     specified_massif_ids=specified_massif_ids)
        plot_name = ''
        if self.window_size_for_smoothing > 1:
            plot_name += ' smoothing values temporally with sliding window of size {}'.format(
                self.window_size_for_smoothing)
        plot_name += '{} top values taken into account for the metric'.format(self.number_of_top_values)
        self.plot_name = plot_name
        self.show_or_save_to_file()

    def visualize_max_graphs_poster(self, massif_name, altitude, snow_abbreviation, color,
                                    label=None, last_plot=True, ax=None, linestyle=None,
                                    tight_pad=None, dpi=None, linewidth=5,
                                    legend_size=None):
        massif_names = self.study.study_massif_names
        # Display the graph of the max on top
        if ax is None:
            ax = plt.gca()
        if massif_name is None:
            x, y = self.study.ordered_years, self.study.observations_annual_maxima.df_maxima_gev.mean(axis=0)
        else:
            x, y = self.smooth_maxima_x_y(massif_names.index(massif_name))
        ax.plot(x, y, color=color, linewidth=linewidth, label=label, linestyle=linestyle)
        # ax.set_ylabel('{} (in {})'.format(snow_abbreviation, self.study.variable_unit), color=color, fontsize=15)

        ax.xaxis.set_ticks(x[1::10])
        ax.tick_params(axis='both', which='major', labelsize=13)
        plot_name = 'Annual maxima of {} in {} at {}m'.format(snow_abbreviation, massif_name, altitude)
        self.plot_name = plot_name
        fontsize = 15
        ax.set_ylabel('{} ({})'.format(snow_abbreviation, self.study.variable_unit), fontsize=fontsize)
        ax.set_xlabel('years', fontsize=fontsize)
        if label is None and massif_name is not None:
            ax.set_title('{} at {} m'.format(massif_name, altitude))
        if last_plot:
            if legend_size is None:
                ax.legend()
            else:
                ax.legend(prop={'size': legend_size})


            self.show_or_save_to_file(add_classic_title=False, no_title=True,
                                      tight_layout=True, tight_pad=tight_pad,
                                      dpi=dpi)
            ax.clear()

    @staticmethod
    def round_sig(x, sig=2):
        return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

    def visualize_gev_graphs_poster(self, massif_name, altitude, snow_abbreviation, color):
        massif_names = self.study.study_massif_names
        # Display the graph of the max on top
        ax = plt.gca()
        _, y = self.smooth_maxima_x_y(massif_names.index(massif_name))
        gev_param = fitted_stationary_gev(x_gev=y)
        # Round up

        # d = {k: self.round_sig(v, 2) for k, v in d.items()}
        # print(d)
        # gev_param = GevParams.from_dict(d)
        x_gev = np.linspace(0.0, 1.5 * max(y), num=1000)
        y_gev = [gev_param.density(x) for x in x_gev]
        ax.plot(x_gev, y_gev, color=color, linewidth=5)
        ax.set_xlabel('y = annual maxima of {} (in {})'.format(snow_abbreviation, self.study.variable_unit),
                      color=color, fontsize=15)
        ax.set_ylabel('$f_{GEV}' + '(y|\mu={},\sigma={},\zeta={})$'.format(*gev_param.to_array()), fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=13)

        # self.visualize_massif_graphs(self.visualize_mean_and_max_graph,
        #                              specified_massif_ids=specified_massif_ids)
        plot_name = 'Gev annual maxima of {} in {} at {}m'.format(snow_abbreviation, massif_name, altitude)
        self.plot_name = plot_name
        self.show_or_save_to_file(add_classic_title=False, no_title=True)
        ax.clear()

    def visualize_mean_and_max_graph(self, ax, massif_id):
        # Display the graph of the max on top
        color_maxima = 'r'
        x, y = self.smooth_maxima_x_y(massif_id)
        ax2 = ax.twinx()
        ax2.plot(x, y, color=color_maxima)
        ax2.set_ylabel('maxima', color=color_maxima)
        # Display the mean graph
        # Counting the sum of 3-consecutive days of snowfall does not have any physical meaning,
        # as we are counting twice some days
        color_mean = 'g'
        tuples_x_y = [(year, np.mean(data[:, massif_id])) for year, data in
                      self.study.year_to_daily_time_serie_array.items()]
        x, y = list(zip(*tuples_x_y))
        x, y = self.average_smoothing_with_sliding_window(x, y,
                                                          window_size_for_smoothing=self.window_size_for_smoothing)
        ax.plot(x, y, color=color_mean)
        ax.set_ylabel('mean'.format(self.window_size_for_smoothing), color=color_mean)
        massif_name = self.study.study_massif_names[massif_id]
        title = massif_name
        title += ' {}={}-{}'.format(*[round(e, 1) for e in list(self.massif_name_to_first_detailed_score[massif_name])])
        ax.set_title(title)
        ax.xaxis.set_ticks(x[2::20])

    def smooth_maxima_x_y(self, massif_id):
        if massif_id not in self.massif_id_to_smooth_maxima:
            tuples_x_y = [(year, annual_maxima[massif_id]) for year, annual_maxima in
                          self.study.year_to_annual_maxima.items()]
            x, y = list(zip(*tuples_x_y))
            x, y = self.average_smoothing_with_sliding_window(x, y,
                                                              window_size_for_smoothing=self.window_size_for_smoothing)
            self.massif_id_to_smooth_maxima[massif_id] = (x, y)
        return self.massif_id_to_smooth_maxima[massif_id]

    def visualize_linear_margin_fit(self, only_first_max_stable=False):
        margin_class = LinearNonStationaryLocationMarginModel if self.temporal_non_stationarity else LinearStationaryMarginModel
        plot_name = 'Full Likelihood with Linear marginals and max stable dependency structure'
        plot_name += '\n(with {} covariance structure when a covariance is needed)'.format(
            str(self.default_covariance_function).split('.')[-1])
        self.plot_name = plot_name

        # Load max stable models
        max_stable_models = load_test_max_stable_models(default_covariance_function=self.default_covariance_function)
        if only_first_max_stable:
            # Keep only the BrownResnick model
            max_stable_models = max_stable_models[1:2]
        if only_first_max_stable is None:
            max_stable_models = []

        # Load axes (either a 2D or 3D array depending on self.coordinates)
        nb_models = len(max_stable_models) + 1
        nb_summary_names = GevParams.NB_SUMMARY_NAMES
        if self.temporal_non_stationarity:
            nb_visualization_times_steps = AbstractMarginFunction.VISUALIZATION_TEMPORAL_STEPS
            # Create one plot for each max stable models
            axes = []
            for _ in range(nb_models):
                axes.append(create_adjusted_axes(nb_rows=nb_summary_names, nb_columns=nb_visualization_times_steps,
                                                 figsize=self.figsize, subplot_space=self.subplot_space))
            # todo: add a fake vizu step at the end, where I could add the independent margin !!
            # rajouter une colonne poru chaque plot, et donner cette colonne à independent margin
        else:
            axes = create_adjusted_axes(nb_rows=nb_models + 1, nb_columns=nb_summary_names,
                                        figsize=self.figsize, subplot_space=self.subplot_space)
            # Plot the margin fit independently on the additional row
            self.visualize_independent_margin_fits(threshold=None, axes=axes[-1], show=False)

        # Plot the smooth margin only
        margin_model = margin_class(coordinates=self.coordinates, starting_point=None)
        estimator = LinearMarginEstimator(dataset=self.dataset, margin_model=margin_model)
        self.fit_and_visualize_estimator(estimator, axes[0], title='without max stable')

        # Plot the smooth margin fitted with a max stable
        for i, max_stable_model in enumerate(max_stable_models, 1):
            margin_model = margin_class(coordinates=self.coordinates, starting_point=None)
            estimator = FullEstimatorInASingleStepWithSmoothMargin(self.dataset, margin_model, max_stable_model)
            title = get_display_name_from_object_type(type(max_stable_model))
            self.fit_and_visualize_estimator(estimator, axes[i], title=title)
        # Add the label
        self.show_or_save_to_file()

    def fit_and_visualize_estimator(self, estimator, axes=None, title=None):
        estimator.fit()

        # Set visualization attributes for margin_fct
        margin_fct = estimator.margin_function_from_fit
        margin_fct._visualization_x_limits = self.study.visualization_x_limits
        margin_fct._visualization_y_limits = self.study.visualization_y_limits
        margin_fct.mask_2D = self.study.mask_french_alps
        if self.temporal_non_stationarity:
            margin_fct.add_future_temporal_steps = True

        axes = margin_fct.visualize_function(show=False, axes=axes, title='')  # type: np.ndarray

        if axes.ndim == 1:
            self.visualize_contour_and_move_axes_limits(axes)
            self.clean_axes_write_title_on_the_left(axes, title)
        else:
            axes = np.transpose(axes)
            for temporal_step, axes_line in zip(margin_fct.temporal_steps, axes):
                self.visualize_contour_and_move_axes_limits(axes_line)
                self.clean_axes_write_title_on_the_left(axes_line, str(temporal_step) + title, left_border=False)

    def visualize_contour_and_move_axes_limits(self, axes):
        for ax in axes:
            self.study.visualize_study(ax, fill=False, show=False)

    @staticmethod
    def clean_axes_write_title_on_the_left(axes, title, left_border=True):
        if left_border is None:
            clean_axes = axes
            ax0 = axes[0]
        elif left_border:
            ax0, *clean_axes = axes
        else:
            *clean_axes, ax0 = axes
        for ax in clean_axes:
            StudyVisualizer.clean_ax(ax)
        ax0.get_yaxis().set_visible(True)
        sub_title = ax0.yaxis.get_label()
        full_title = title + '\n\n' + sub_title._text
        label_function = ax0.set_ylabel if left_border or left_border is None else ax0.set_xlabel
        label_function(full_title)
        ax0.tick_params(axis=u'both', which=u'both', length=0)

    @staticmethod
    def clean_ax(ax):
        ax.tick_params(axis=u'both', which=u'both', length=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')

    def show_or_save_to_file(self, add_classic_title=False, no_title=True, tight_layout=False, tight_pad=None,
                             dpi=None, folder_for_variable=True, plot_name=None):
        if plot_name is not None:
            self.plot_name = plot_name

        if isinstance(self.study, AbstractAdamontStudy):
            prefix = gcm_rcm_couple_to_str(self.study.gcm_rcm_couple)
            prefix = prefix.replace('/', '-')
            self.plot_name = prefix + ' ' + self.plot_name

        assert self.plot_name is not None
        if add_classic_title:
            title = self.study.title
            title += '\n' + self.plot_name
        else:
            title = self.plot_name
        if self.only_one_graph:
            plt.suptitle(self.plot_name,  y=1.0)
        elif not no_title:
            plt.suptitle(title,  y=1.0)
        if self.show:
            plt.show()
        if self.save_to_file:
            main_title, specific_title = '_'.join(self.study.title.split()).split('/')
            main_title += self.study.season_name
            if folder_for_variable:
                filename = "{}/{}/".format(VERSION_TIME, main_title)
            else:
                filename = "{}/".format(VERSION_TIME)
            if not self.only_one_graph:
                filename += "{}".format('_'.join(self.plot_name.split())) + '_'
            filename += specific_title
            # Save a first time in transparent
            self.savefig_in_results(filename, transparent=True)
            self.savefig_in_results(filename, transparent=False, tight_pad=tight_pad)

    @classmethod
    def savefig_in_results(cls, filename, transparent=True, tight_pad=None):
        img_format = 'svg' if transparent else 'png'
        filepath = op.join(AbstractStudy.result_full_path, filename + '.' + img_format)
        if transparent:
            dir_list = filepath.split('/')
            dir_list[-1:] = ['transparent', dir_list[-1]]
            filepath = '/'.join(dir_list)
        dirname = op.dirname(filepath)
        if not op.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        if transparent:
            plt.savefig(filepath, bbox_inches=0, transparent=True)
        else:
            if tight_pad is not None:
                plt.tight_layout(**tight_pad)
            else:
                plt.tight_layout()
            plt.savefig(filepath, bbox_inches=0, transparent=False)

            
        # if dpi is not None:
        #     plt.savefig(filepath, dpi=dpi)
        # else:
        #     plt.savefig(filepath)

    def visualize_independent_margin_fits(self, threshold=None, axes=None, show=True):
        # Fit either a GEV or a GPD
        if threshold is None:
            params_names = GevParams.SUMMARY_NAMES
            df = self.df_gev_mle
        else:
            params_names = GpdParams.SUMMARY_NAMES
            df = self.df_gpd_mle(threshold)

        if axes is None:
            fig, axes = plt.subplots(1, len(params_names))
            fig.subplots_adjust(hspace=self.subplot_space, wspace=self.subplot_space)

        for i, param_name in enumerate(params_names):
            ax = axes[i]
            self.study.visualize_study(ax=ax, massif_name_to_value=df.loc[param_name, :].to_dict(), show=False,
                                       replace_blue_by_white=param_name != GevParams.SHAPE,
                                       label=param_name)
        self.clean_axes_write_title_on_the_left(axes, title='Independent fits')

        if show:
            plt.show()

    def visualize_summary_of_annual_values_and_stationary_gev_fit(self):
        fig, axes = plt.subplots(3, 4)
        fig.subplots_adjust(hspace=self.subplot_space, wspace=self.subplot_space)

        # 1) First row, some experimental indicator
        axes_first_row = axes[0]
        df_maxima_gev = self.df_maxima_gev
        name_to_serie = OrderedDict()
        name_to_serie['mean'] = df_maxima_gev.mean(axis=1)
        name_to_serie['std'] = df_maxima_gev.std(axis=1)
        name_to_serie['quantile 0.9'] = df_maxima_gev.quantile(q=0.9, axis=1)
        name_to_serie['quantile 0.99'] = df_maxima_gev.quantile(q=0.99, axis=1)
        for (name, serie), ax in zip(name_to_serie.items(), axes_first_row):
            self.study.visualize_study(ax=ax,
                                       massif_name_to_value=serie.to_dict(),
                                       show=False,
                                       label='empirical ' + name)

        # 2) Second row, gev parameters fitted independently (and a qqplot)
        axes_second_row = axes[1]
        for ax, param_name in zip(axes_second_row, GevParams.PARAM_NAMES):
            self.study.visualize_study(ax=ax,
                                       massif_name_to_value=self.df_gev_parameters.loc[param_name, :].to_dict(),
                                       show=False,
                                       replace_blue_by_white=param_name != GevParams.SHAPE,
                                       label=param_name)
        # todo: add qqplot drawn for each massif on the map in the last cell
        # or just it could be some fitting score based on the qqplot... and we just display the value
        # like the log likelihood, (or we could also display some uncertainty here)

        # 3) Third row, gev indicator
        axes_third_row = axes[2]
        for ax, indicator_name in zip(axes_third_row, GevParams.indicator_names()):
            self.study.visualize_study(ax=ax,
                                       massif_name_to_value=self.df_gev_indicators.loc[indicator_name, :].to_dict(),
                                       show=False,
                                       label='gev ' + indicator_name)

        # Clean all ax
        for ax in axes.flatten():
            StudyVisualizer.clean_ax(ax)
        self.plot_name = 'Overview of empirical and stationary gev indicators'
        self.show_or_save_to_file()

    def visualize_annual_mean_values(self, ax=None, take_mean_value=True):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=self.figsize)

        massif_name_to_value = OrderedDict()
        df_annual_total = self.study.df_annual_total
        for massif_name in self.study.study_massif_names:
            # We take the mean over all the annual values, otherwise we take the max
            value = df_annual_total.loc[:, massif_name]
            value = value.mean() if take_mean_value else value.max()
            massif_name_to_value[massif_name] = value
        print(len(massif_name_to_value))
        print(massif_name_to_value)
        self.study.visualize_study(ax=ax, massif_name_to_value=massif_name_to_value, show=self.show, add_text=True,
                                   label=self.study.variable_name)

    """ Statistics methods """

    @property
    def df_maxima_gev(self) -> pd.DataFrame:
        return self.study.observations_annual_maxima.df_maxima_gev

    @cached_property
    def massif_name_to_gev_mle_fitted(self) -> Dict[str, GevParams]:
        return {massif_name: fitted_stationary_gev(self.df_maxima_gev.loc[massif_name])
                for massif_name in self.study.study_massif_names}

    @cached_property
    def df_gev_parameters(self) -> pd.DataFrame:
        massif_to_gev_mle = {massif_name: gev_params.to_dict()
                             for massif_name, gev_params in self.massif_name_to_gev_mle_fitted.items()}
        return pd.DataFrame(massif_to_gev_mle, columns=self.study.study_massif_names)

    @cached_property
    def df_gev_indicators(self) -> pd.DataFrame:
        massif_to_gev_mle = {massif_name: gev_params.indicator_name_to_value
                             for massif_name, gev_params in self.massif_name_to_gev_mle_fitted.items()}
        return pd.DataFrame(massif_to_gev_mle, columns=self.study.study_massif_names)

    @property
    def df_all_daily_time_series_concatenated(self) -> pd.DataFrame:
        df_list = [pd.DataFrame(time_serie, columns=self.study.study_massif_names) for time_serie in
                   self.study.year_to_daily_time_serie_array.values()]
        df_concatenated = pd.concat(df_list)
        return df_concatenated

    def df_gpd_mle(self, threshold) -> pd.DataFrame:
        # Fit a margin fit on each massif
        massif_to_gev_mle = {massif_name: GpdMleFit(self.df_all_daily_time_series_concatenated[massif_name],
                                                    threshold=threshold).gpd_params.summary_serie
                             for massif_name in self.study.study_massif_names}
        return pd.DataFrame(massif_to_gev_mle, columns=self.study.study_massif_names)

    @staticmethod
    def average_smoothing_with_sliding_window(x, y, window_size_for_smoothing):
        # Average on windows of size 2*M+1 (M elements on each side)
        kernel = np.ones(window_size_for_smoothing) / window_size_for_smoothing
        y = np.convolve(y, kernel, mode='valid')
        assert window_size_for_smoothing % 2 == 1
        if window_size_for_smoothing > 1:
            nb_to_delete = int(window_size_for_smoothing // 2)
            x = np.array(x)[nb_to_delete:-nb_to_delete]
        assert len(x) == len(y), "{} vs {}".format(len(x), len(y))
        return x, y

    # PLot functions that should be common

    def plot_map(self, cmap, graduation, label, massif_name_to_value, plot_name, add_x_label=True,
                 negative_and_positive_values=True, massif_name_to_text=None, altitude=None, add_colorbar=True,
                 max_abs_change=None, xlabel=None, fontsize_label=10, massif_names_with_white_dot=None):
        if altitude is None:
            altitude = self.study.altitude
        if len(massif_name_to_value) > 0:
            load_plot(cmap, graduation, label, massif_name_to_value, altitude,
                      add_x_label=add_x_label, negative_and_positive_values=negative_and_positive_values,
                      massif_name_to_text=massif_name_to_text,
                      add_colorbar=add_colorbar, max_abs_change=max_abs_change, xlabel=xlabel,
                      fontsize_label=fontsize_label, massif_names_with_white_dot=massif_names_with_white_dot)
            self.plot_name = plot_name
            # self.show_or_save_to_file(add_classic_title=False, tight_layout=True, no_title=True, dpi=500)
            self.show_or_save_to_file(add_classic_title=False, no_title=True, dpi=500, tight_layout=True)
            plt.close()

    def plot_abstract(self, massif_name_to_value, label, plot_name, fit_method='', graduation=10.0, cmap=plt.cm.bwr,
                      add_x_label=True, negative_and_positive_values=True, massif_name_to_text=None):
        # Regroup the plot by altitudes
        plot_name1 = '{}/{}'.format(self.study.altitude, plot_name)
        # Regroup the plot by type of plot also
        plot_name2 = '{}/{}'.format(plot_name.split()[0], plot_name)
        for plot_name in [plot_name1, plot_name2]:
            self.plot_map(cmap, graduation, label, massif_name_to_value, plot_name, add_x_label, negative_and_positive_values,
                          massif_name_to_text, )

