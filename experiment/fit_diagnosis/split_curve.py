import numpy as np
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import seaborn as sns

from typing import Union, List

from extreme_estimator.estimator.full_estimator import AbstractFullEstimator
from extreme_estimator.estimator.margin_estimator import AbstractMarginEstimator
from extreme_estimator.extreme_models.margin_model.margin_function.combined_margin_function import \
    CombinedMarginFunction
from extreme_estimator.extreme_models.margin_model.margin_function.utils import error_dict_between_margin_functions
from extreme_estimator.gev_params import GevParams
from spatio_temporal_dataset.dataset.simulation_dataset import FullSimulatedDataset
from spatio_temporal_dataset.slicer.split import Split, ALL_SPLITS_EXCEPT_ALL


class SplitCurve(object):

    def __init__(self, nb_fit: int = 1):
        self.nb_fit = nb_fit
        self.margin_function_fitted_all = None

    def fit(self, show=True):
        self.margin_function_fitted_all = []

        for i in range(self.nb_fit):
            # A new dataset with the same margin, but just the observations are resampled
            self.dataset = self.load_dataset()
            # Both split must be defined
            assert not self.dataset.slicer.some_required_ind_are_not_defined
            self.margin_function_sample = self.dataset.margin_model.margin_function_sample

            print('Fitting {}/{}...'.format(i + 1, self.nb_fit))
            self.estimator = self.load_estimator(self.dataset)
            # Fit the estimator and get the margin_function
            self.estimator.fit()
            self.margin_function_fitted_all.append(self.estimator.margin_function_fitted)

        # Individual error dict
        self.error_dict_all = [error_dict_between_margin_functions(self.margin_function_sample, m)
                               for m in self.margin_function_fitted_all]

        # Mean margin
        self.mean_margin_function_fitted = CombinedMarginFunction.from_margin_functions(self.margin_function_fitted_all)
        self.mean_error_dict = error_dict_between_margin_functions(self.margin_function_sample,
                                                                   self.mean_margin_function_fitted)

        if show:
            self.visualize()

    def load_dataset(self):
        pass

    def load_estimator(self, dataset):
        pass

    @property
    def main_title(self):
        return self.dataset.slicer.summary(show=False)

    def visualize(self):
        fig, axes = plt.subplots(len(GevParams.GEV_VALUE_NAMES), 2)
        fig.subplots_adjust(hspace=0.4, wspace=0.4, )
        for i, gev_value_name in enumerate(GevParams.GEV_VALUE_NAMES):
            self.margin_graph(axes[i, 0], gev_value_name)
            self.score_graph(axes[i, 1], gev_value_name)
        fig.suptitle(self.main_title)
        plt.show()

    def margin_graph(self, ax, gev_value_name):
        # Create bins of data, each with an associated color corresponding to its error

        data = self.mean_error_dict[gev_value_name].values
        nb_bins = 10
        limits = np.linspace(data.min(), data.max(), num=nb_bins + 1)
        limits[-1] += 0.01
        colors = cm.binary(limits)

        # Display train/test points
        for split, marker in [(self.dataset.train_split, 'o'), (self.dataset.test_split, 'x')]:
            for left_limit, right_limit, color in zip(limits[:-1], limits[1:], colors):
                # Find for the split the index
                data_ind = self.mean_error_dict[gev_value_name].loc[
                    self.dataset.coordinates.coordinates_index(split)].values
                data_filter = np.logical_and(left_limit <= data_ind, data_ind < right_limit)

                self.margin_function_sample.set_datapoint_display_parameters(split, datapoint_marker=marker,
                                                                             filter=data_filter, color=color)
                self.margin_function_sample.visualize_single_param(gev_value_name, ax, show=False)

        # Display the individual fitted curve
        self.mean_margin_function_fitted.color = 'lightskyblue'
        for m in self.margin_function_fitted_all:
            m.visualize_single_param(gev_value_name, ax, show=False)
        # Display the mean fitted curve
        self.mean_margin_function_fitted.color = 'blue'
        self.mean_margin_function_fitted.visualize_single_param(gev_value_name, ax, show=False)

    def score_graph(self, ax, gev_value_name):
        # todo: for the moment only the train/test are interresting (the spatio temporal isn"t working yet)

        sns.set_style('whitegrid')
        s = self.mean_error_dict[gev_value_name]
        for split in self.dataset.splits:
            ind = self.dataset.coordinates_index(split)
            data = s.loc[ind].values
            sns.kdeplot(data, bw=0.5, ax=ax, label=split.name).set(xlim=0)
        ax.legend()
        # X axis
        ax.set_xlabel('Absolute error in percentage')
        plt.setp(ax.get_xticklabels(), visible=True)
        ax.xaxis.set_tick_params(labelbottom=True)
        # Y axis
        ax.set_ylabel(gev_value_name)
        plt.setp(ax.get_yticklabels(), visible=True)
        ax.yaxis.set_tick_params(labelbottom=True)
