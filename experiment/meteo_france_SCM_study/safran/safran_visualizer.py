import math
import os
import os.path as op

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiment.meteo_france_SCM_study.abstract_study import AbstractStudy
from experiment.utils import average_smoothing_with_sliding_window
from extreme_estimator.estimator.full_estimator.abstract_full_estimator import \
    FullEstimatorInASingleStepWithSmoothMargin
from extreme_estimator.estimator.margin_estimator.abstract_margin_estimator import SmoothMarginEstimator
from extreme_estimator.extreme_models.margin_model.smooth_margin_model import LinearAllParametersAllDimsMarginModel
from extreme_estimator.extreme_models.max_stable_model.abstract_max_stable_model import CovarianceFunction, \
    AbstractMaxStableModelWithCovarianceFunction
from extreme_estimator.margin_fits.abstract_params import AbstractParams
from extreme_estimator.margin_fits.gev.gev_params import GevParams
from extreme_estimator.margin_fits.gev.gevmle_fit import GevMleFit
from extreme_estimator.margin_fits.gpd.gpd_params import GpdParams
from extreme_estimator.margin_fits.gpd.gpdmle_fit import GpdMleFit
from extreme_estimator.margin_fits.plot.create_shifted_cmap import get_color_rbga_shifted
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from test.test_utils import load_test_max_stable_models
from utils import get_display_name_from_object_type, VERSION_TIME, float_to_str_with_only_some_significant_digits


class StudyVisualizer(object):

    def __init__(self, study: AbstractStudy, show=True, save_to_file=False, only_one_graph=False, only_first_row=False,
                 vertical_kde_plot=False, year_for_kde_plot=None):
        self.only_first_row = only_first_row
        self.only_one_graph = only_one_graph
        self.save_to_file = save_to_file
        self.study = study
        self.plot_name = None
        # KDE PLOT ARGUMENTS
        self.vertical_kde_plot=vertical_kde_plot
        self.year_for_kde_plot = year_for_kde_plot
        self.show = False if self.save_to_file else show
        self.window_size_for_smoothing = 21
        if self.only_one_graph:
            self.figsize = (6.0, 4.0)
        elif self.only_first_row:
            self.figsize = (8.0, 6.0)
        else:
            self.figsize = (16.0, 10.0)

    @property
    def observations(self):
        return self.study.observations_annual_maxima

    @property
    def coordinates(self):
        return self.study.massifs_coordinates

    @property
    def dataset(self):
        return AbstractDataset(self.observations, self.coordinates)

    # Graph for each massif / or groups of massifs

    def visualize_massif_graphs(self, visualize_function):
        if self.only_one_graph:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            visualize_function(ax, 0)
        else:
            nb_columns = 5
            nb_rows = 1 if self.only_first_row else math.ceil(len(self.study.safran_massif_names) / nb_columns)
            fig, axes = plt.subplots(nb_rows, nb_columns, figsize=self.figsize)
            fig.subplots_adjust(hspace=1.0, wspace=1.0)
            if self.only_first_row:
                for massif_id, massif_name in enumerate(self.study.safran_massif_names[:nb_columns]):
                    ax = axes[massif_id]
                    visualize_function(ax, massif_id)
            else:
                for massif_id, massif_name in enumerate(self.study.safran_massif_names):
                    row_id, column_id = massif_id // nb_columns, massif_id % nb_columns
                    ax = axes[row_id, column_id]
                    visualize_function(ax, massif_id)

    def visualize_all_experimental_law(self):
        self.visualize_massif_graphs(self.visualize_experimental_law)
        self.plot_name = ' Empirical distribution '
        self.plot_name += 'with all available data' if self.year_for_kde_plot is None else \
            'for the year {}'.format(self.year_for_kde_plot)
        self.show_or_save_to_file()

    def visualize_experimental_law(self, ax, massif_id):
        # Display the experimental law for a given massif
        if self.year_for_kde_plot is not None:
            all_massif_data = self.study.year_to_daily_time_serie[self.year_for_kde_plot][:, massif_id]
        else:
            all_massif_data = np.concatenate([data[:, massif_id] for data in self.study.year_to_daily_time_serie.values()])
        all_massif_data = np.sort(all_massif_data)

        # Kde plot, and retrieve the data forming the line
        color_kde = 'b'
        sns.kdeplot(all_massif_data, bw=1, ax=ax, color=color_kde, vertical=self.vertical_kde_plot).set(xlim=0)
        data_x, data_y = ax.lines[0].get_data()

        # Plot the mean point in green
        x_level_to_color = {
            np.mean(all_massif_data): ('g', 'mean'),
        }
        # Plot some specific quantiles in their color
        for p, color, name in zip(AbstractParams.QUANTILE_P_VALUES, AbstractParams.QUANTILE_COLORS, AbstractParams.QUANTILE_NAMES):
            x_level = all_massif_data[int(p * len(all_massif_data))]
            x_level_to_color[x_level] = (color, name)

        for xi, (color, name) in x_level_to_color.items():
            if self.vertical_kde_plot:
                yi = xi
                xi = np.interp(yi, data_y, data_x)
            else:
                yi = np.interp(xi, data_x, data_y)
            ax.scatter([xi], [yi], color=color, marker="o", label=name)

        label_function = ax.set_xlabel if self.vertical_kde_plot else ax.set_ylabel
        label_function('Probability Density function f(x)', color=color_kde)

        xlabel = 'x = {}'.format(self.study.title) if self.only_one_graph else 'x'
        label_function = ax.set_ylabel if self.vertical_kde_plot else ax.set_xlabel
        label_function(xlabel)
        extraticks = [float(float_to_str_with_only_some_significant_digits(x, nb_digits=2))
                      for x in sorted(list(x_level_to_color.keys()))]
        if not self.only_one_graph:
            extraticks = [extraticks[0], extraticks[-1]]
        set_ticks_function = ax.set_yticks if self.vertical_kde_plot else ax.set_xticks
        set_ticks_function(extraticks)
        if not self.only_one_graph:
            ax.set_title(self.study.safran_massif_names[massif_id])
        ax.legend()

    def visualize_all_mean_and_max_graphs(self):
        self.visualize_massif_graphs(self.visualize_mean_and_max_graph)
        self.plot_name = ' mean with sliding window of size {}'.format(self.window_size_for_smoothing)
        self.show_or_save_to_file()

    def visualize_mean_and_max_graph(self, ax, massif_id):
        # Display the graph of the max on top
        color_maxima = 'r'
        tuples_x_y = [(year, annual_maxima[massif_id]) for year, annual_maxima in
                      self.study.year_to_annual_maxima.items()]
        x, y = list(zip(*tuples_x_y))
        ax2 = ax.twinx()
        ax2.plot(x, y, color=color_maxima)
        ax2.set_ylabel('maxima', color=color_maxima)
        # Display the mean graph
        # Counting the sum of 3-consecutive days of snowfall does not have any physical meaning,
        # as we are counting twice some days
        color_mean = 'g'
        tuples_x_y = [(year, np.mean(data[:, massif_id])) for year, data in self.study.year_to_daily_time_serie.items()]
        x, y = list(zip(*tuples_x_y))
        x, y = average_smoothing_with_sliding_window(x, y, window_size_for_smoothing=self.window_size_for_smoothing)
        ax.plot(x, y, color=color_mean)
        ax.set_ylabel('mean with sliding window of size {}'.format(self.window_size_for_smoothing), color=color_mean)
        ax.set_xlabel('year')
        ax.set_title(self.study.safran_massif_names[massif_id])

    def visualize_linear_margin_fit(self, only_first_max_stable=False):
        self.plot_name = 'Full Likelihood with Linear marginals and max stable dependency structure'
        default_covariance_function = CovarianceFunction.cauchy
        max_stable_models = load_test_max_stable_models(default_covariance_function=default_covariance_function)
        if only_first_max_stable:
            max_stable_models = max_stable_models[:1]
        fig, axes = plt.subplots(len(max_stable_models) + 1, len(GevParams.SUMMARY_NAMES), figsize=self.figsize)
        fig.subplots_adjust(hspace=1.0, wspace=1.0)
        margin_class = LinearAllParametersAllDimsMarginModel
        # Plot the smooth margin only
        margin_model = margin_class(coordinates=self.coordinates)
        estimator = SmoothMarginEstimator(dataset=self.dataset, margin_model=margin_model)
        self.fit_and_visualize_estimator(estimator, axes[0], title='without max stable')
        # Plot the smooth margin fitted with a max stable
        for i, max_stable_model in enumerate(max_stable_models, 1):
            margin_model = margin_class(coordinates=self.coordinates)
            estimator = FullEstimatorInASingleStepWithSmoothMargin(self.dataset, margin_model, max_stable_model)
            title = get_display_name_from_object_type(type(max_stable_model))
            if isinstance(max_stable_model, AbstractMaxStableModelWithCovarianceFunction):
                title += ' ' + str(default_covariance_function).split('.')[-1]
            self.fit_and_visualize_estimator(estimator, axes[i], title=title)
        self.show_or_save_to_file()

    def fit_and_visualize_estimator(self, estimator, axes=None, title=None):
        estimator.fit()
        axes = estimator.margin_function_fitted.visualize_function(show=False, axes=axes, title=title)
        for ax in axes:
            self.study.visualize(ax, fill=False, show=False)

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
            filename = "{}/{}".format(VERSION_TIME, '_'.join(self.study.title.split()))
            if not self.only_one_graph:
                filename += "/{}".format('_'.join(self.plot_name.split()))
            filepath = op.join(self.study.result_full_path, filename + '.png')
            dir = op.dirname(filepath)
            if not op.exists(dir):
                os.makedirs(dir, exist_ok=True)
            plt.savefig(filepath)

    def visualize_independent_margin_fits(self, threshold=None, axes=None):
        if threshold is None:
            params_names = GevParams.SUMMARY_NAMES
            df = self.df_gev_mle_each_massif
            # todo: understand how Maurienne could be negative
            # print(df.head())
        else:
            params_names = GpdParams.SUMMARY_NAMES
            df = self.df_gpd_mle_each_massif(threshold)

        if axes is None:
            fig, axes = plt.subplots(1, len(params_names))
            fig.subplots_adjust(hspace=1.0, wspace=1.0)

        for i, gev_param_name in enumerate(params_names):
            ax = axes[i]
            massif_name_to_value = df.loc[gev_param_name, :].to_dict()
            # Compute the middle point of the values for the color map
            values = list(massif_name_to_value.values())
            colors = get_color_rbga_shifted(ax, gev_param_name, values)
            massif_name_to_fill_kwargs = {massif_name: {'color': color} for massif_name, color in
                                          zip(self.study.safran_massif_names, colors)}
            self.study.visualize(ax=ax, massif_name_to_fill_kwargs=massif_name_to_fill_kwargs, show=False)

        if self.show:
            plt.show()

    def visualize_cmap(self, massif_name_to_value):
        orig_cmap = plt.cm. coolwarm
        # shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0.75, name='shifted')

        massif_name_to_fill_kwargs = {massif_name: {'color': orig_cmap(value)} for massif_name, value in
                                      massif_name_to_value.items()}

        self.study.visualize(massif_name_to_fill_kwargs=massif_name_to_fill_kwargs)

    """ Statistics methods """

    @property
    def df_gev_mle_each_massif(self):
        # Fit a margin_fits on each massif
        massif_to_gev_mle = {
            massif_name: GevMleFit(self.study.observations_annual_maxima.loc[massif_name]).gev_params.summary_serie
            for massif_name in self.study.safran_massif_names}
        return pd.DataFrame(massif_to_gev_mle, columns=self.study.safran_massif_names)

    def df_gpd_mle_each_massif(self, threshold):
        # Fit a margin fit on each massif
        massif_to_gev_mle = {massif_name: GpdMleFit(self.study.df_all_snowfall_concatenated[massif_name],
                                                    threshold=threshold).gpd_params.summary_serie
                             for massif_name in self.study.safran_massif_names}
        return pd.DataFrame(massif_to_gev_mle, columns=self.study.safran_massif_names)
