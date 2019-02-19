from typing import Dict

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from extreme_estimator.margin_fits.plot.shifted_color_map import shiftedColorMap
from extreme_estimator.margin_fits.extreme_params import ExtremeParams
from extreme_estimator.margin_fits.gev.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.slicer.split import Split


def plot_extreme_param(ax, gev_param_name, values):
    # Load the shifted cmap to center on a middle point
    vmin, vmax = np.min(values), np.max(values)
    cmap = [plt.cm.coolwarm, plt.cm.bwr, plt.cm.seismic][1]
    if gev_param_name == ExtremeParams.SHAPE and vmin < 0:
        midpoint = 1 - vmax / (vmax + abs(vmin))
        shifted_cmap = shiftedColorMap(cmap, midpoint=midpoint, name='shifted')
    else:
        shifted_cmap = shiftedColorMap(cmap, midpoint=0.0, name='shifted')
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = cbar.ColorbarBase(cax, cmap=shifted_cmap, norm=norm)
    cb.set_label(gev_param_name)
    return norm, shifted_cmap


def get_color_rbga_shifted(ax, gev_param_name, values):
    """
    For some display it was necessary to transform dark blue values into white values
    """
    norm, shifted_cmap = plot_extreme_param(ax, gev_param_name, values)
    m = cm.ScalarMappable(norm=norm, cmap=shifted_cmap)
    colors = [m.to_rgba(value) for value in values]
    if gev_param_name != ExtremeParams.SHAPE:
        colors = [color if color[2] == 1 else (1, 1, 1, 1) for color in colors]
    return colors


def imshow_shifted(ax, gev_param_name, values, x, y):
    norm, shifted_cmap = plot_extreme_param(ax, gev_param_name, values)
    shifted_cmap.set_bad(color='white')

    masked_array = values
    if gev_param_name != ExtremeParams.SHAPE:
        epsilon = 1.0
        value = np.min(values)
        # The right blue corner will be blue (but most of the time, another display will be on top)
        masked_array[-1, -1] = value - epsilon

    ax.imshow(masked_array, extent=(x.min(), x.max(), y.min(), y.max()), cmap=shifted_cmap)

