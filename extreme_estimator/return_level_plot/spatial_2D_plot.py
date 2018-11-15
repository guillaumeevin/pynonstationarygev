from itertools import product
from typing import List, Dict

import matplotlib.pyplot as plt

from extreme_estimator.extreme_models.margin_model.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from extreme_estimator.gev_params import GevParams

plt.style.use('seaborn-white')


class Spatial2DPlot(object):

    def __init__(self, name_to_margin_function: Dict[str, AbstractMarginFunction]):
        self.name_to_margin_function = name_to_margin_function # type: Dict[str, AbstractMarginFunction]
        self.grid_columns = GevParams.GEV_PARAM_NAMES

    def plot(self):
        nb_grid_rows, nb_grid_columns = len(self.name_to_margin_function), len(self.grid_columns)
        fig, axes = plt.subplots(nb_grid_rows, nb_grid_columns, sharex='col', sharey='row')
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        margin_function: AbstractMarginFunction
        for i, (name, margin_function) in enumerate(self.name_to_margin_function.items()):
            for j, param_name in enumerate(self.grid_columns):
                ax = axes[i, j] if nb_grid_rows > 1 else axes[j]
                margin_function.visualize_2D(gev_param_name=param_name, ax=ax)
                ax.set_title("{} for {}".format(param_name, name))
        fig.suptitle('Spatial2DPlot')
        plt.show()
