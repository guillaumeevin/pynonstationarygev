import matplotlib
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from extreme_fit.distribution.abstract_params import AbstractParams


def get_shifted_map(vmin, vmax, cmap=plt.cm.bwr):
    # Load the shifted cmap to center on a middle point
    if vmin < 0 < vmax:
        midpoint = 1 - vmax / (vmax + abs(vmin))
    elif vmin < 0 and vmax < 0:
        midpoint = 1.0
    elif vmin > 0 and vmax > 0:
        midpoint = 0.0
    else:
        raise ValueError('Unexpected values: vmin={}, vmax={}'.format(vmin, vmax))
    # cmap = [plt.cm.coolwarm, plt.cm.bwr, plt.cm.seismic][1]
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
        cb.set_label(label, fontsize=15)
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



# from: https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib/20528097
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap
