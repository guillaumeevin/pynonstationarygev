import os
import os.path as op
import pickle

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from extreme_estimator.estimator.abstract_estimator import AbstractEstimator
from extreme_estimator.extreme_models.margin_model.margin_function.combined_margin_function import \
    CombinedMarginFunction
from extreme_estimator.extreme_models.margin_model.margin_function.utils import error_dict_between_margin_functions
from extreme_estimator.gev_params import GevParams
from spatio_temporal_dataset.dataset.abstract_dataset import get_subset_dataset
from spatio_temporal_dataset.dataset.simulation_dataset import SimulatedDataset
from utils import get_full_path, get_display_name_from_object_type

SIMULATION_RELATIVE_PATH = op.join('local', 'simulation')


class AbstractSimulation(object):

    def __init__(self):
        self.nb_fit = 1
        self.margin_function_fitted_list = None
        self.full_dataset = None
        self.error_dict_all = None
        self.margin_function_sample = None
        self.mean_error_dict = None
        self.mean_margin_function_fitted = None

    def fit(self, estimator_class, show=True):
        assert estimator_class not in self.already_fitted_estimator_names, \
            'This estimator class has already been fitted.' \
            'Create a child class, if you wish to change some default parameters'

        # Load full dataset
        full_dataset = self.load_full_dataset()
        assert len(full_dataset.subset_id_to_column_idxs) == self.nb_fit
        assert not full_dataset.slicer.some_required_ind_are_not_defined

        # Fit a margin function on each subset
        margin_function_fitted_list = []
        for subset_id in range(self.nb_fit):
            print('Fitting {}/{} of {}...'.format(subset_id + 1, self.nb_fit,
                                                  get_display_name_from_object_type(estimator_class)))
            dataset = get_subset_dataset(full_dataset, subset_id=subset_id)  # type: SimulatedDataset
            estimator = estimator_class.from_dataset(dataset)  # type: AbstractEstimator
            # Fit the estimator and get the margin_function
            estimator.fit()
            margin_function_fitted_list.append(estimator.margin_function_fitted)

        # Individual error dict
        self.dump_fitted_margins_pickle(estimator_class, margin_function_fitted_list)

        if show:
            self.visualize_comparison_graph(estimator_names=[estimator_class])

    def dump_fitted_margins_pickle(self, estimator_class, margin_function_fitted_list):
        with open(self.fitted_margins_pickle_path(estimator_class), 'wb') as fp:
            pickle.dump(margin_function_fitted_list, fp)

    def load_fitted_margins_pickles(self, estimator_class):
        with open(self.fitted_margins_pickle_path(estimator_class), 'rb') as fp:
            return pickle.load(fp)

    def visualize_comparison_graph(self, estimator_names=None):
        # Visualize the result of several estimators on the same graph
        if estimator_names is None:
            estimator_names = self.already_fitted_estimator_names
        assert len(estimator_names) > 0
        # Load dataset
        self.full_dataset = self.load_full_dataset()
        self.margin_function_sample = self.full_dataset.margin_model.margin_function_sample

        fig, axes = self.load_fig_and_axes()
        for estimator_name in estimator_names:
            self.margin_function_fitted_list = self.load_fitted_margins_pickles(estimator_name)

            self.error_dict_all = [error_dict_between_margin_functions(reference=self.margin_function_sample,
                                                                       fitted=margin_function_fitted)
                                   for margin_function_fitted in self.margin_function_fitted_list]

            # Mean margin
            self.mean_margin_function_fitted = CombinedMarginFunction.from_margin_functions(
                self.margin_function_fitted_list)

            self.mean_error_dict = error_dict_between_margin_functions(self.margin_function_sample,
                                                                       self.mean_margin_function_fitted)
            self.visualize(fig, axes, show=False)
        plt.show()

    @property
    def already_fitted_estimator_names(self):
        return [d for d in os.listdir(self.directory_path) if op.isdir(op.join(self.directory_path, d))]

    @property
    def main_title(self):
        return self.full_dataset.slicer.summary(show=False)

    @staticmethod
    def load_fig_and_axes():
        fig, axes = plt.subplots(len(GevParams.GEV_VALUE_NAMES), 2)
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        return fig, axes

    def visualize(self, fig=None, axes=None, show=True):
        if fig is None or axes is None:
            fig, axes = self.load_fig_and_axes()
        for i, gev_value_name in enumerate(GevParams.GEV_VALUE_NAMES):
            self.margin_graph(axes[i, 0], gev_value_name)
            self.score_graph(axes[i, 1], gev_value_name)
        if show:
            fig.suptitle(self.main_title)
            plt.show()

    def margin_graph(self, ax, gev_value_name):
        # Create bins of data, each with an associated color corresponding to its error

        data = self.mean_error_dict[gev_value_name].values
        data_min, data_max = data.min(), data.max()
        nb_bins = 10
        limits = np.linspace(data_min, data_max, num=nb_bins + 1)
        limits[-1] += 0.01
        # Binary color should
        colors = cm.binary((limits - data_min / (data_max - data_min)))

        # Display train/test points
        for split, marker in [(self.full_dataset.train_split, 'o'), (self.full_dataset.test_split, 'x')]:
            for left_limit, right_limit, color in zip(limits[:-1], limits[1:], colors):
                # Find for the split the index
                data_ind = self.mean_error_dict[gev_value_name].loc[
                    self.full_dataset.coordinates.coordinates_index(split)].values
                data_filter = np.logical_and(left_limit <= data_ind, data_ind < right_limit)

                self.margin_function_sample.set_datapoint_display_parameters(split, datapoint_marker=marker,
                                                                             filter=data_filter, color=color)
                self.margin_function_sample.visualize_single_param(gev_value_name, ax, show=False)

        # Display the individual fitted curve
        self.mean_margin_function_fitted.color = 'lightskyblue'
        for m in self.margin_function_fitted_list:
            m.visualize_single_param(gev_value_name, ax, show=False)
        # Display the mean fitted curve
        self.mean_margin_function_fitted.color = 'blue'
        self.mean_margin_function_fitted.visualize_single_param(gev_value_name, ax, show=False)

    def score_graph(self, ax, gev_value_name):
        # todo: for the moment only the train/test are interresting (the spatio temporal isn"t working yet)

        sns.set_style('whitegrid')
        s = self.mean_error_dict[gev_value_name]
        for split in self.full_dataset.splits:
            ind = self.full_dataset.coordinates_index(split)
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

    # Input/Output

    @property
    def name(self):
        return str(type(self)).split('.')[-1].split('Simulation')[0]

    @property
    def directory_path(self):
        return op.join(get_full_path(relative_path=SIMULATION_RELATIVE_PATH), self.name, str(self.nb_fit))

    @property
    def dataset_path(self):
        return op.join(self.directory_path, 'dataset')

    @property
    def dataset_csv_path(self):
        return self.dataset_path + '.csv'

    @property
    def dataset_pickle_path(self):
        return self.dataset_path + '.pkl'

    def fitted_margins_pickle_path(self, estimator_class):
        d = op.join(self.directory_path, get_display_name_from_object_type(estimator_class))
        if not op.exists(d):
            os.makedirs(d, exist_ok=True)
        return op.join(d, 'fitted_margins.pkl')

    def dump(self):
        pass

    def _dump(self, dataset: SimulatedDataset):
        dataset.create_subsets(nb_subsets=self.nb_fit)
        dataset.to_csv(self.dataset_csv_path)
        # Pickle Dataset
        if op.exists(self.dataset_pickle_path):
            print('A dataset already exists, we will keep it intact, delete it manually if you want to change it')
        else:
            with open(self.dataset_pickle_path, 'wb') as fp:
                pickle.dump(dataset, fp)

    def load_full_dataset(self) -> SimulatedDataset:
        # Class to handle pickle loading (and in case of module refactoring, I could change the module name here)
        class RenamingUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'mymodule':
                    module = 'mymodule2'
                return super().find_class(module, name)

        with open(self.dataset_pickle_path, 'rb') as fp:
            dataset = RenamingUnpickler(fp).load()
        return dataset
