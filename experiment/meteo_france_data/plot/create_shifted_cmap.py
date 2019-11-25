import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from experiment.meteo_france_data.plot.shifted_color_map import shiftedColorMap
from extreme_fit.distribution.abstract_params import AbstractParams


def get_shifted_map(vmin, vmax):
    # Load the shifted cmap to center on a middle point
    if vmin < 0 < vmax:
        midpoint = 1 - vmax / (vmax + abs(vmin))
    elif vmin < 0 and vmax < 0:
        midpoint = 1.0
    elif vmin > 0 and vmax > 0:
        midpoint = 0.0
    else:
        raise ValueError('Unexpected values: vmin={}, vmax={}'.format(vmin, vmax))
    cmap = [plt.cm.coolwarm, plt.cm.bwr, plt.cm.seismic][1]
    shifted_cmap = shiftedColorMap(cmap, midpoint=midpoint, name='shifted')
    return shifted_cmap


def create_colorbase_axis(ax, label, cmap, norm, ticks_values_and_labels=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.0)
    ticks = ticks_values_and_labels[0] if ticks_values_and_labels is not None else None
    cb = cbar.ColorbarBase(cax, cmap=cmap, norm=norm, ticks=ticks)
    if ticks_values_and_labels is not None:
        cb.ax.set_yticklabels([str(t) for t in ticks_values_and_labels[1]])
    if isinstance(label, str):
        cb.set_label(label)
    return norm


def get_norm(vmin, vmax):
    return mpl.colors.Normalize(vmin=vmin, vmax=vmax)


def get_colors(values, cmap, vmin, vmax, replace_blue_by_white=False):
    norm = get_norm(vmin, vmax)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = [m.to_rgba(value) for value in values]
    if replace_blue_by_white:
        colors = [color if color[2] != 1 else (1, 1, 1, 1) for color in colors]
    return colors


def imshow_shifted(ax, gev_param_name, values, visualization_extend, mask_2D=None):
    condition = np.isnan(values)
    if mask_2D is not None:
        condition |= mask_2D
    masked_array = np.ma.masked_where(condition, values)
    vmin, vmax = np.min(masked_array), np.max(masked_array)
    shifted_cmap = get_shifted_map(vmin, vmax)
    norm = get_norm(vmin, vmax)
    create_colorbase_axis(ax, gev_param_name, shifted_cmap, norm)
    shifted_cmap.set_bad(color='white')
    if gev_param_name != AbstractParams.SHAPE:
        epsilon = 1e-2 * (np.max(values) - np.min(values))
        value = np.min(values)
        # The right blue corner will be blue (but most of the time, another display will be on top)
        masked_array[-1, -1] = value - epsilon
    # IMPORTANT: Origin for all the plots is at the bottom left corner
    ax.imshow(masked_array, extent=visualization_extend, cmap=shifted_cmap, origin='lower')
