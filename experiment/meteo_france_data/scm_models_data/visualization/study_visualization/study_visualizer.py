import os
import os.path as op
from collections import OrderedDict
from random import sample, seed
from typing import Dict

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiment.meteo_france_data.scm_models_data.abstract_extended_study import AbstractExtendedStudy
from experiment.trend_analysis.abstract_score import MeanScore, AbstractTrendScore
from experiment.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from experiment.trend_analysis.univariate_test.abstract_univariate_test import AbstractUnivariateTest
from experiment.trend_analysis.non_stationary_trends import \
    ConditionalIndedendenceLocationTrendTest, MaxStableLocationTrendTest, IndependenceLocationTrendTest
from experiment.meteo_france_data.scm_models_data.visualization.utils import create_adjusted_axes
from experiment.trend_analysis.univariate_test.utils import compute_gev_change_point_test_results
from experiment.utils import average_smoothing_with_sliding_window
from extreme_fit.distribution.abstract_params import AbstractParams
from extreme_fit.estimator.full_estimator.abstract_full_estimator import \
    FullEstimatorInASingleStepWithSmoothMargin
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearNonStationaryLocationMarginModel, \
    LinearStationaryMarginModel
from extreme_fit.model.margin_model.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from extreme_fit.model.margin_model.param_function.param_function import AbstractParamFunction
from extreme_fit.model.max_stable_model.abstract_max_stable_model import CovarianceFunction
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.distribution.gev.ismev_gev_fit import IsmevGevFit
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
from utils import get_display_name_from_object_type, VERSION_TIME, float_to_str_with_only_some_significant_digits, \
    cached_property

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
                 complete_non_stationary_trend_analysis=False, normalization_under_one_observations=True,
                 score_class=MeanScore):
        super().__init__(save_to_file, only_one_graph, only_first_row, show)
        self.nb_cores = 7
        self.massif_id_to_smooth_maxima = {}
        self.temporal_non_stationarity = temporal_non_stationarity
        self.only_first_row = only_first_row
        self.only_one_graph = only_one_graph
        self.save_to_file = save_to_file
        self.study = study
        self.plot_name = None

        self.normalization_under_one_observations = normalization_under_one_observations
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
        self.score_class = score_class
        self.score = self.score_class(self.number_of_top_values)  # type: AbstractTrendScore

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
                if self.normalization_under_one_observations:
                    self._observations.normalize()
                if self.verbose:
                    self._observations.print_summary()
        return self._observations

    def observation_massif_id(self, massif_id):
        annual_maxima = [data[massif_id] for data in self.study.year_to_annual_maxima.values()]
        df_annual_maxima = pd.DataFrame(annual_maxima, index=self.temporal_coordinates.index)
        observation_massif_id = AnnualMaxima(df_maxima_gev=df_annual_maxima)
        if self.normalization_under_one_observations:
            observation_massif_id.normalize()
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

    # TEMPORAL TREND

    def visualize_all_independent_temporal_trend(self):
        massifs_ids = [self.study.study_massif_names.index(name) for name in self.specified_massif_names_median_scores]
        self.visualize_massif_graphs(self.visualize_independent_temporal_trend, specified_massif_ids=massifs_ids)
        self.plot_name = ' Independent temporal trend \n'
        self.show_or_save_to_file()

    def visualize_independent_temporal_trend(self, ax, massif_id):
        assert self.temporal_non_stationarity
        # Create a dataset with temporal coordinates from the massif id
        dataset_massif_id = AbstractDataset(self.observation_massif_id(massif_id), self.temporal_coordinates)
        trend_test = IndependenceLocationTrendTest(station_name=self.study.study_massif_names[massif_id],
                                                   dataset=dataset_massif_id, verbose=self.verbose,
                                                   multiprocessing=self.multiprocessing)
        trend_test.visualize(ax, self.complete_non_stationary_trend_analysis)

    def visualize_temporal_trend_relevance(self):
        assert self.temporal_non_stationarity
        trend_tests = [ConditionalIndedendenceLocationTrendTest(dataset=self.dataset, verbose=self.verbose,
                                                                multiprocessing=self.multiprocessing)]

        max_stable_models = load_test_max_stable_models(default_covariance_function=self.default_covariance_function)
        for max_stable_model in [max_stable_models[1], max_stable_models[-2]]:
            trend_tests.append(MaxStableLocationTrendTest(dataset=self.dataset,
                                                          max_stable_model=max_stable_model, verbose=self.verbose,
                                                          multiprocessing=self.multiprocessing))

        nb_trend_tests = len(trend_tests)
        fig, axes = plt.subplots(1, nb_trend_tests, figsize=self.figsize)
        if nb_trend_tests == 1:
            axes = [axes]
        fig.subplots_adjust(hspace=self.subplot_space, wspace=self.subplot_space)
        for ax, trend_test in zip(axes, trend_tests):
            trend_test.visualize(ax, complete_analysis=self.complete_non_stationary_trend_analysis)

        plot_name = 'trend tests'
        plot_name += ' with {} applied spatially & temporally'.format(
            get_display_name_from_object_type(self.transformation_class))
        if self.normalization_under_one_observations:
            plot_name += '(and maxima <= 1)'
        self.plot_name = plot_name
        self.show_or_save_to_file()

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

    @cached_property
    def massif_name_to_detailed_scores(self):
        """
        This score respect the following property.
        Between two successive score, then if the starting year was neither a top10 maxima nor a top10 minima,
        then the score will not change

        The following case for instance gives a constant score wrt to the starting year
        because all the maxima and all the minima are at the end
            smooth_maxima = [0 for _ in years]
            smooth_maxima[-20:-10] = [i for i in range(10)]
            smooth_maxima[-10:] = [-i for i in range(10)]
        :return:
        """
        # Ordered massif by scores
        massif_name_to_scores = {}
        for massif_id, massif_name in enumerate(self.study.study_massif_names):
            years, smooth_maxima = self.smooth_maxima_x_y(massif_id)
            detailed_scores = []
            for j, starting_year in enumerate(self.starting_years):
                detailed_scores.append(self.score.get_detailed_score(years, smooth_maxima))
                assert years[0] == starting_year, "{} {}".format(years[0], starting_year)
                # Remove the first element from the list
                years = years[1:]
                smooth_maxima = smooth_maxima[1:]
            massif_name_to_scores[massif_name] = np.array(detailed_scores)
        return massif_name_to_scores

    def massif_name_to_df_trend_type(self, trend_test_class, starting_year_to_weight):
        """
        Create a DataFrame with massif as index
        :param trend_test_class:
        :param starting_year_to_weight:
        :return:
        """
        massif_name_to_df_trend_type = {}
        for massif_id, massif_name in enumerate(self.study.study_massif_names):
            trend_type_and_weight = []
            years, smooth_maxima = self.smooth_maxima_x_y(massif_id)
            for starting_year, weight in starting_year_to_weight.items():
                test_trend_type = self.compute_trend_test_result(smooth_maxima, starting_year, trend_test_class, years)
                trend_type_and_weight.append((test_trend_type, weight))
            df = pd.DataFrame(trend_type_and_weight, columns=['trend type', 'weight'])
            massif_name_to_df_trend_type[massif_name] = df
        return massif_name_to_df_trend_type

    def massif_name_to_gev_change_point_test_results(self, trend_test_class_for_change_point_test,
                                                     starting_years_for_change_point_test,
                                                     nb_massif_for_change_point_test=None,
                                                     sample_one_massif_from_each_region=True):
        if self.trend_test_class_for_change_point_test is None:
            # Set the attribute is not already done
            self.trend_test_class_for_change_point_test = trend_test_class_for_change_point_test
            self.starting_years_for_change_point_test = starting_years_for_change_point_test
            self.nb_massif_for_change_point_test = nb_massif_for_change_point_test
            self.sample_one_massif_from_each_region = sample_one_massif_from_each_region
        else:
            # Check that the argument are the same
            assert self.trend_test_class_for_change_point_test == trend_test_class_for_change_point_test
            assert self.starting_years == starting_years_for_change_point_test
            assert self.nb_massif_for_change_point_test == nb_massif_for_change_point_test
            assert self.sample_one_massif_from_each_region == sample_one_massif_from_each_region

        return self._massif_name_to_gev_change_point_test_results

    @cached_property
    def _massif_name_to_gev_change_point_test_results(self):
        massif_name_to_gev_change_point_test_results = {}
        if self.nb_massif_for_change_point_test is None:
            massif_names = self.study.study_massif_names
        else:
            # Set the random seed to the same number so that
            print('Setting the random seed to ensure similar sampling in the fast mode')
            seed(42)
            if self.sample_one_massif_from_each_region:
                # Get one massif from each region to ensure that the fast plot will not crash
                assert self.nb_massif_for_change_point_test >= 4, 'we need at least one massif from each region'
                massif_names = [AbstractExtendedStudy.region_name_to_massif_names[r][0]
                                for r in AbstractExtendedStudy.real_region_names]
                massif_names_for_sampling = list(set(self.study.study_massif_names) - set(massif_names))
                nb_massif_for_sampling = self.nb_massif_for_change_point_test - len(AbstractExtendedStudy.real_region_names)
                massif_names += sample(massif_names_for_sampling, k=nb_massif_for_sampling)
            else:
                massif_names = sample(self.study.study_massif_names, k=self.nb_massif_for_change_point_test)

        for massif_id, massif_name in enumerate(massif_names):
            years, smooth_maxima = self.smooth_maxima_x_y(massif_id)
            gev_change_point_test_results = compute_gev_change_point_test_results(self.multiprocessing, smooth_maxima,
                                                                                  self.starting_years_for_change_point_test,
                                                                                  self.trend_test_class_for_change_point_test,
                                                                                  years)
            massif_name_to_gev_change_point_test_results[massif_name] = gev_change_point_test_results
        return massif_name_to_gev_change_point_test_results

    def df_trend_spatio_temporal(self, trend_test_class_for_change_point_test,
                                 starting_years_for_change_point_test,
                                 nb_massif_for_change_point_test=None,
                                 sample_one_massif_from_each_region=True):
        """
        Index are the massif
        Columns are the starting year

        :param trend_test_class:
        :param starting_year_to_weight:
        :return:
        """
        massif_name_to_trend_res = {}
        massif_name_to_gev_change_point_test_results = self.massif_name_to_gev_change_point_test_results(
            trend_test_class_for_change_point_test,
            starting_years_for_change_point_test,
            nb_massif_for_change_point_test,
            sample_one_massif_from_each_region)
        for massif_name, gev_change_point_test_results in massif_name_to_gev_change_point_test_results.items():
            trend_test_res, best_idxs = gev_change_point_test_results
            trend_test_res = [(a, b, c, d, e, f) if i in best_idxs else (np.nan, np.nan, c, np.nan, np.nan, np.nan)
                              for i, (a, b, c, d, e, f, *_) in enumerate(trend_test_res)]
            massif_name_to_trend_res[massif_name] = list(zip(*trend_test_res))
        nb_res = len(list(massif_name_to_trend_res.values())[0])
        assert nb_res == 6

        all_massif_name_to_res = [{k: v[idx_res] for k, v in massif_name_to_trend_res.items()}
                                  for idx_res in range(nb_res)]
        return [pd.DataFrame(massif_name_to_res, index=self.starting_years_for_change_point_test).transpose()
                for massif_name_to_res in all_massif_name_to_res]

    @staticmethod
    def compute_trend_test_result(smooth_maxima, starting_year, trend_test_class, years):
        trend_test = trend_test_class(years, smooth_maxima, starting_year)  # type: AbstractUnivariateTest
        return trend_test.test_trend_type

    def df_trend_test_count(self, trend_test_class, starting_year_to_weight):
        """
        Index are the trend type
        Columns are the massif

        :param starting_year_to_weight:
        :param trend_test_class:
        :return:
        """
        massif_name_to_df_trend_type = self.massif_name_to_df_trend_type(trend_test_class, starting_year_to_weight)
        df = pd.concat([100 * v.groupby(['trend type']).sum()
                        for v in massif_name_to_df_trend_type.values()], axis=1, sort=False)
        df.fillna(0.0, inplace=True)
        assert np.allclose(df.sum(axis=0), 100)
        # Add the significant trend into the count of normal trend
        if AbstractUnivariateTest.SIGNIFICATIVE_POSITIVE_TREND in df.index:
            df.loc[AbstractUnivariateTest.POSITIVE_TREND] += df.loc[AbstractUnivariateTest.SIGNIFICATIVE_POSITIVE_TREND]
        if AbstractUnivariateTest.SIGNIFICATIVE_NEGATIVE_TREND in df.index:
            df.loc[AbstractUnivariateTest.NEGATIVE_TREND] += df.loc[AbstractUnivariateTest.SIGNIFICATIVE_NEGATIVE_TREND]
        return df

    @cached_property
    def massif_name_to_scores(self):
        return {k: v[:, 0] for k, v in self.massif_name_to_detailed_scores.items()}

    @cached_property
    def massif_name_to_first_detailed_score(self):
        return {k: v[0] for k, v in self.massif_name_to_detailed_scores.items()}

    @cached_property
    def massif_name_to_first_score(self):
        return {k: v[0] for k, v in self.massif_name_to_scores.items()}

    @property
    def specified_massif_names_median_scores(self):
        return sorted(self.study.study_massif_names, key=lambda s: np.median(self.massif_name_to_scores[s]))

    @property
    def specified_massif_names_first_score(self):
        return sorted(self.study.study_massif_names, key=lambda s: self.massif_name_to_scores[s][0])

    def visualize_all_score_wrt_starting_year(self):
        specified_massif_names = self.specified_massif_names_median_scores
        # Add one graph at the end
        specified_massif_names += [None]
        self.visualize_massif_graphs(self.visualize_score_wrt_starting_year,
                                     specified_massif_ids=specified_massif_names)
        plot_name = ''
        plot_name += '{} top values for each score, abscisse represents starting year for a trend'.format(
            self.number_of_top_values)
        self.plot_name = plot_name
        self.show_or_save_to_file()

    def visualize_score_wrt_starting_year(self, ax, massif_name):
        if massif_name is None:
            percentage, title = self.percentages_of_negative_trends()
            scores = percentage
            ax.set_ylabel('% of negative trends')
            # Add two lines of interest
            years_of_interest = [1963, 1976]
            colors = ['g', 'r']
            for year_interest, color in zip(years_of_interest, colors):
                ax.axvline(x=year_interest, color=color)
                year_score = scores[self.starting_years.index(year_interest)]
                ax.axhline(y=year_score, color=color)
        else:
            ax.set_ylabel(get_display_name_from_object_type(self.score))
            scores = self.massif_name_to_scores[massif_name]
            title = massif_name
        ax.plot(self.starting_years, scores)
        ax.set_title(title)
        ax.xaxis.set_ticks(self.starting_years[2::20])

    def percentages_of_negative_trends(self):
        print('start computing percentages negative trends')
        # scores = np.median([np.array(v) < 0 for v in self.massif_name_to_scores.values()], axis=0)
        # Take the mean with respect to the massifs
        # We obtain an array whose length equal the length of starting years
        scores = np.mean([np.array(v) < 0 for v in self.massif_name_to_scores.values()], axis=0)
        percentages = 100 * scores
        # First argmin, first argmax
        argmin, argmax = np.argmin(scores), np.argmax(scores)
        # Last argmin, last argmax
        # argmin, argmax = len(scores) - 1 - np.argmin(scores[::-1]), len(scores) - 1 - np.argmax(scores[::-1])
        top_starting_year_for_positive_trend = self.starting_years[argmin]
        top_starting_year_for_negative_trend = self.starting_years[argmax]
        top_percentage_positive_trend = round(100 - percentages[argmin], 0)
        top_percentage_negative_trend = round(percentages[argmax], 0)
        title = "Global trend; > 0: {}% in {}; < 0: {}% in {}".format(top_percentage_positive_trend,
                                                                      top_starting_year_for_positive_trend,
                                                                      top_percentage_negative_trend,
                                                                      top_starting_year_for_negative_trend)

        return percentages, title

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

    def visualize_max_graphs_poster(self, massif_name, altitude, snow_abbreviation, color):
        massif_names = self.study.study_massif_names
        # Display the graph of the max on top
        ax = plt.gca()
        x, y = self.smooth_maxima_x_y(massif_names.index(massif_name))
        ax.plot(x, y, color=color, linewidth=5)
        ax.set_ylabel('{} (in {})'.format(snow_abbreviation, self.study.variable_unit), color=color, fontsize=15)
        ax.xaxis.set_ticks(x[2::10])
        ax.tick_params(axis='both', which='major', labelsize=13)

        # self.visualize_massif_graphs(self.visualize_mean_and_max_graph,
        #                              specified_massif_ids=specified_massif_ids)
        plot_name = 'Annual maxima of {} in {} at {}m'.format(snow_abbreviation, massif_name, altitude)
        self.plot_name = plot_name
        self.show_or_save_to_file(add_classic_title=False, no_title=True)
        ax.clear()

    @staticmethod
    def round_sig(x, sig=2):
        return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

    def visualize_gev_graphs_poster(self, massif_name, altitude, snow_abbreviation, color):
        massif_names = self.study.study_massif_names
        # Display the graph of the max on top
        ax = plt.gca()
        _, y = self.smooth_maxima_x_y(massif_names.index(massif_name))
        gev_param = IsmevGevFit(x_gev=y).gev_params
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
        x, y = average_smoothing_with_sliding_window(x, y, window_size_for_smoothing=self.window_size_for_smoothing)
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

    def show_or_save_to_file(self, add_classic_title=True, no_title=False, tight_layout=False):
        if tight_layout:
            plt.tight_layout()
        assert self.plot_name is not None
        if add_classic_title:
            title = self.study.title
            title += '\n' + self.plot_name
        else:
            title = self.plot_name
        if self.only_one_graph:
            plt.suptitle(self.plot_name)
        elif not no_title:
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
        for ax, gev_param_name in zip(axes_second_row, GevParams.PARAM_NAMES):
            self.study.visualize_study(ax=ax,
                                       massif_name_to_value=self.df_gev_parameters.loc[gev_param_name, :].to_dict(),
                                       show=False,
                                       replace_blue_by_white=gev_param_name != GevParams.SHAPE,
                                       label=gev_param_name)
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
        return {massif_name: IsmevGevFit(self.df_maxima_gev.loc[massif_name]).gev_params
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
