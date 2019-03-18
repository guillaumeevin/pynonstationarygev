import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from extreme_estimator.margin_fits.extreme_params import ExtremeParams
from extreme_estimator.margin_fits.plot.shifted_color_map import shiftedColorMap


def plot_extreme_param(ax, label: str, values: np.ndarray):
    # Load the shifted cmap to center on a middle point
    vmin, vmax = np.min(values), np.max(values)
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
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.03)
    cb = cbar.ColorbarBase(cax, cmap=shifted_cmap, norm=norm)
    cb.set_label(label)
    return norm, shifted_cmap


def get_color_rbga_shifted(ax, replace_blue_by_white: bool, values: np.ndarray, label=None):
    """
    For some display it was necessary to transform dark blue values into white values
    """
    norm, shifted_cmap = plot_extreme_param(ax, label, values)
    m = cm.ScalarMappable(norm=norm, cmap=shifted_cmap)
    colors = [m.to_rgba(value) for value in values]
    # We do not want any blue values for parameters other than the Shape
    # So when the value corresponding to the blue color is 1, then we set the color to white, i.e. (1,1,1,1)
    if replace_blue_by_white:
        colors = [color if color[2] != 1 else (1, 1, 1, 1) for color in colors]
    return colors


def imshow_shifted(ax, gev_param_name, values, x, y):
    masked_array = np.ma.masked_where(np.isnan(values), values)
    norm, shifted_cmap = plot_extreme_param(ax, gev_param_name, masked_array)
    shifted_cmap.set_bad(color='white')
    if gev_param_name != ExtremeParams.SHAPE:
        epsilon = 1e-2 * (np.max(values) - np.min(values))
        value = np.min(values)
        # The right blue corner will be blue (but most of the time, another display will be on top)
        masked_array[-1, -1] = value - epsilon
    # IMPORTANT: Origin for all the plots is at the bottom left corner
    ax.imshow(masked_array, extent=(x.min(), x.max(), y.min(), y.max()), cmap=shifted_cmap, origin='lower')

