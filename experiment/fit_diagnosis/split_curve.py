import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from typing import Union, List

from extreme_estimator.estimator.full_estimator import AbstractFullEstimator
from extreme_estimator.estimator.margin_estimator import AbstractMarginEstimator
from extreme_estimator.extreme_models.margin_model.margin_function.utils import error_dict_between_margin_functions
from extreme_estimator.gev_params import GevParams
from spatio_temporal_dataset.dataset.simulation_dataset import FullSimulatedDataset
from spatio_temporal_dataset.slicer.split import Split, ALL_SPLITS_EXCEPT_ALL


class MarginFunction(object):

    def margin_function(self, gev_param: GevParams) -> float:
        pass


class LocFunction(MarginFunction):

    def margin_function(self, gev_param: GevParams) -> float:
        return gev_param.location


class SplitCurve(object):

    def __init__(self, dataset: FullSimulatedDataset, estimator: Union[AbstractFullEstimator, AbstractMarginEstimator],
                 margin_functions: List[MarginFunction]):
        # Dataset is already loaded and will not be modified
        self.dataset = dataset
        # Both split must be defined
        assert not self.dataset.slicer.some_required_ind_are_not_defined
        self.margin_function_sample = self.dataset.margin_model.margin_function_sample

        self.estimator = estimator
        # Fit the estimator and get the margin_function
        self.estimator.fit()
        # todo: potentially I will do the fit several times, and retrieve the mean error
        # there is a big variablility so it would be really interesting to average, to make real
        self.margin_function_fitted = estimator.margin_function_fitted

        self.error_dict = error_dict_between_margin_functions(self.margin_function_sample, self.margin_function_fitted)

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
        # Display the fitted curve
        self.margin_function_fitted.visualize_single_param(gev_value_name, ax, show=False)
        # Display train/test points
        for split, marker in [(self.dataset.train_split, 'o'), (self.dataset.test_split, 'x')]:
            self.margin_function_sample.set_datapoint_display_parameters(split, datapoint_marker=marker)
            self.margin_function_sample.visualize_single_param(gev_value_name, ax, show=False)

    def score_graph(self, ax, gev_value_name):
        # todo: for the moment only the train/test are interresting (the spatio temporal isn"t working yet)
        sns.set_style('whitegrid')
        s = self.error_dict[gev_value_name]
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
