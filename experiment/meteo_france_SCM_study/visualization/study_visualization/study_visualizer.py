import os
import os.path as op
from collections import OrderedDict
from typing import Union

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiment.meteo_france_SCM_study.abstract_study import AbstractStudy
from experiment.meteo_france_SCM_study.visualization.study_visualization.non_stationary_trends import \
    ConditionalIndedendenceLocationTrendTest, MaxStableLocationTrendTest
from experiment.meteo_france_SCM_study.visualization.utils import create_adjusted_axes
from experiment.utils import average_smoothing_with_sliding_window
from extreme_estimator.estimator.full_estimator.abstract_full_estimator import \
    FullEstimatorInASingleStepWithSmoothMargin
from extreme_estimator.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_estimator.extreme_models.margin_model.linear_margin_model import LinearNonStationaryLocationMarginModel, \
    LinearStationaryMarginModel
from extreme_estimator.extreme_models.margin_model.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from extreme_estimator.extreme_models.margin_model.param_function.param_function import AbstractParamFunction
from extreme_estimator.extreme_models.max_stable_model.abstract_max_stable_model import CovarianceFunction
from extreme_estimator.margin_fits.abstract_params import AbstractParams
from extreme_estimator.margin_fits.gev.gev_params import GevParams
from extreme_estimator.margin_fits.gev.gevmle_fit import GevMleFit
from extreme_estimator.margin_fits.gpd.gpd_params import GpdParams
from extreme_estimator.margin_fits.gpd.gpdmle_fit import GpdMleFit
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.abstract_spatio_temporal_coordinates import \
    AbstractSpatioTemporalCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformed_coordinates import TransformedCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from test.test_utils import load_test_max_stable_models
from utils import get_display_name_from_object_type, VERSION_TIME, float_to_str_with_only_some_significant_digits

BLOCK_MAXIMA_DISPLAY_NAME = 'block maxima '


class StudyVisualizer(object):

    def __init__(self, study: AbstractStudy, show=True, save_to_file=False, only_one_graph=False, only_first_row=False,
                 vertical_kde_plot=False, year_for_kde_plot=None, plot_block_maxima_quantiles=False,
                 temporal_non_stationarity=False,
                 transformation_class=None,
                 normalization_under_one_observations=True):
        self.massif_id_to_smooth_maxima = {}
        self.temporal_non_stationarity = temporal_non_stationarity
        self.only_first_row = only_first_row
        self.only_one_graph = only_one_graph
        self.save_to_file = save_to_file
        self.study = study
        self.plot_name = None

        self.normalization_under_one_observations = normalization_under_one_observations
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

        self.window_size_for_smoothing = 21

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

        # Modify some class attributes
        # Remove some assert
        AbstractParamFunction.OUT_OF_BOUNDS_ASSERT = False
        # INCREASE THE TEMPORAL STEPS FOR VISUALIZATION
        AbstractMarginFunction.VISUALIZATION_TEMPORAL_STEPS = 5

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
                if self.normalization_under_one_observations:
                    self._observations.normalize()
                self._observations.print_summary()
        return self._observations

    # Graph for each massif / or groups of massifs

    def visualize_massif_graphs(self, visualize_function, specified_massif_names=None):
        if self.only_one_graph:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            visualize_function(ax, 0)
        else:
            nb_columns = 5
            nb_rows = 1 if self.only_first_row else math.ceil(len(self.study.study_massif_names) / nb_columns)
            fig, axes = plt.subplots(nb_rows, nb_columns, figsize=self.figsize)
            fig.subplots_adjust(hspace=self.subplot_space, wspace=self.subplot_space)
            if self.only_first_row:
                for massif_id, massif_name in enumerate(self.study.study_massif_names[:nb_columns]):
                    ax = axes[massif_id]
                    visualize_function(ax, massif_id)
            else:
                if specified_massif_names is None:
                    massif_ids = list(range(len(self.study.study_massif_names)))
                else:
                    massif_ids = [self.study.study_massif_names.index(massif_name) for massif_name in specified_massif_names]
                for j, massif_id in enumerate(massif_ids):
                    row_id, column_id = j // nb_columns, j % nb_columns
                    ax = axes[row_id, column_id]
                    visualize_function(ax, massif_id)

    def visualize_all_experimental_law( self):
        self.visualize_massif_graphs(self.visualize_experimental_law)
        self.plot_name = ' Empirical distribution \n'
        self.plot_name += 'with data from the 23 mountain chains of the French Alps ' if self.year_for_kde_plot is None else \
            'for the year {}'.format(self.year_for_kde_plot)
        self.show_or_save_to_file()

    def visualize_temporal_trend_relevance(self, complete_analysis, verbose=True, multiprocessing=False):
        self.temporal_non_stationarity = True
        trend_tests = [ConditionalIndedendenceLocationTrendTest(dataset=self.dataset, verbose=verbose,
                                                                multiprocessing=multiprocessing)]

        max_stable_models = load_test_max_stable_models(default_covariance_function=self.default_covariance_function)
        for max_stable_model in [max_stable_models[1], max_stable_models[-2]]:
            trend_tests.append(MaxStableLocationTrendTest(dataset=self.dataset,
                                                          max_stable_model=max_stable_model, verbose=verbose,
                                                          multiprocessing=multiprocessing))

        nb_trend_tests = len(trend_tests)
        fig, axes = plt.subplots(1, nb_trend_tests, figsize=self.figsize)
        if nb_trend_tests == 1:
            axes = [axes]
        fig.subplots_adjust(hspace=self.subplot_space, wspace=self.subplot_space)
        for ax, trend_test in zip(axes, trend_tests):
            trend_test.visualize(ax, complete_analysis=complete_analysis)

        plot_name = 'trend tests'
        plot_name += ' with {} applied spatially & temporally'.format(get_display_name_from_object_type(self.transformation_class))
        if self.normalization_under_one_observations:
            plot_name += '(and maxima <= 1)'
        self.plot_name = plot_name
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

    def visualize_all_mean_and_max_graphs(self):
        # Compute the order of massif names
        massif_name_to_score = {}
        for massif_id, massif_name in enumerate(self.study.study_massif_names):
            score = self.smooth_maxima_x_y(massif_id)[1].argmax()
            massif_name_to_score[massif_name] = score
        ordered_massif_names = sorted(self.study.study_massif_names[:], key=lambda s: massif_name_to_score[s])
        self.visualize_massif_graphs(self.visualize_mean_and_max_graph, specified_massif_names=ordered_massif_names)
        self.plot_name = ' smoothing values temporally with sliding window of size {}'.format(self.window_size_for_smoothing)
        self.show_or_save_to_file()

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
        x, y = average_smoothing_with_sliding_window(x, y, window_size_for_smoothing=self.window_size_for_smoothing)
        ax.plot(x, y, color=color_mean)
        ax.set_ylabel('mean'.format(self.window_size_for_smoothing), color=color_mean)
        ax.set_xlabel('year')
        ax.set_title(self.study.study_massif_names[massif_id])
        ax.xaxis.set_ticks(x[2::10])


    def smooth_maxima_x_y(self, massif_id):
        if massif_id not in self.massif_id_to_smooth_maxima:
            tuples_x_y = [(year, annual_maxima[massif_id]) for year, annual_maxima in
                          self.study.year_to_annual_maxima.items()]
            x, y = list(zip(*tuples_x_y))
            x, y = average_smoothing_with_sliding_window(x, y, window_size_for_smoothing=self.window_size_for_smoothing)
            self.massif_id_to_smooth_maxima[massif_id] = (x, y)
        return self.massif_id_to_smooth_maxima[massif_id]

    def visualize_brown_resnick_fit(self):
        pass

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
        margin_fct = estimator.margin_function_fitted
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
        if left_border:
            ax0, *clean_axes = axes
        else:
            *clean_axes, ax0 = axes
        for ax in clean_axes:
            ax.tick_params(axis=u'both', which=u'both', length=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
        ax0.get_yaxis().set_visible(True)
        sub_title = ax0.yaxis.get_label()
        full_title = title + '\n\n' + sub_title._text
        label_function = ax0.set_ylabel if left_border else ax0.set_xlabel
        label_function(full_title)
        ax0.tick_params(axis=u'both', which=u'both', length=0)

    def show_or_save_to_file(self):
        assert self.plot_name is not None
        title = self.study.title
        title += '\n' + self.plot_name
        if self.only_one_graph:
            plt.suptitle(self.plot_name)
        else:
            plt.suptitle(title)
        if self.show:
            plt.show()
        if self.save_to_file:
            main_title, specific_title = '_'.join(self.study.title.split()).split('/')
            filename = "{}/{}/".format(VERSION_TIME, main_title)
            if not self.only_one_graph:
                filename += "{}".format('_'.join(self.plot_name.split())) + '_'
            filename += specific_title
            filepath = op.join(self.study.result_full_path, filename + '.png')
            dirname = op.dirname(filepath)
            if not op.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            plt.savefig(filepath)

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

        for i, gev_param_name in enumerate(params_names):
            ax = axes[i]
            self.study.visualize_study(ax=ax, massif_name_to_value=df.loc[gev_param_name, :].to_dict(), show=False,
                                       replace_blue_by_white=gev_param_name != GevParams.SHAPE,
                                       label=gev_param_name)
        self.clean_axes_write_title_on_the_left(axes, title='Independent fits')

        if show:
            plt.show()

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

    @property
    def df_gev_mle(self) -> pd.DataFrame:
        # Fit a margin_fits on each massif
        massif_to_gev_mle = {massif_name: GevMleFit(self.df_maxima_gev.loc[massif_name]).gev_params.summary_serie
                             for massif_name in self.study.study_massif_names}
        return pd.DataFrame(massif_to_gev_mle, columns=self.study.study_massif_names)

    def df_gpd_mle(self, threshold) -> pd.DataFrame:
        # Fit a margin fit on each massif
        massif_to_gev_mle = {massif_name: GpdMleFit(self.study.df_all_daily_time_series_concatenated[massif_name],
                                                    threshold=threshold).gpd_params.summary_serie
                             for massif_name in self.study.study_massif_names}
        return pd.DataFrame(massif_to_gev_mle, columns=self.study.study_massif_names)
