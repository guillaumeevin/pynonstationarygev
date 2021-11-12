import matplotlib
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
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


def remove_the_extreme_colors(cmap, epsilon=0.05):
    epsilon = epsilon
    colors = cmap(np.linspace(epsilon, 1 - epsilon, cmap.N))
    return LinearSegmentedColormap.from_list('without extreme', colors)


def get_inverse_colormap(cmap):
    colors = cmap(np.linspace(1, 0, cmap.N))
    return LinearSegmentedColormap.from_list('Inverse', colors)


def get_cmap_with_inverted_blue_and_green_channels(cmap):
    colors = cmap(np.linspace(0, 1, cmap.N))
    colors = [[c[0], c[2], c[1], c[3]] for c in colors]
    return LinearSegmentedColormap.from_list('Inverted channels', colors)


def get_lower_half_colormap(cmap):
    colors = cmap(np.linspace(0, 0.5, cmap.N // 2))
    # Create a new colormap from those colors
    return LinearSegmentedColormap.from_list('Lower Half', colors)


def get_upper_half_colormap(cmap):
    colors = cmap(np.linspace(0.5, 1, cmap.N // 2))
    # Create a new colormap from those colors
    return LinearSegmentedColormap.from_list('Upper Half', colors)


def get_upper_two_third_colormap(cmap):
    colors = cmap(np.linspace(0.333, 1, int(0.666 * cmap.N)))
    # Create a new colormap from those colors
    return LinearSegmentedColormap.from_list('Upper Half', colors)


def create_colorbase_axis(ax, label, cmap, norm, ticks_values_and_labels=None, fontsize=15, position='right',
                          rescale_ticks=False, snowfall=True,
                          vmin=None, vmax=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size='5%', pad=0.0)
    ticks = ticks_values_and_labels[0] if ticks_values_and_labels is not None else None
    horizontal_plot = position == 'top'
    if rescale_ticks is True:
        if vmin is None:
            vmin = norm.vmin
        if vmax is None:
            vmax = norm.vmax
        factor = max(abs(vmin), abs(vmax))
        if vmin < 0 < vmax:
            ticks = [t * (norm.vmax - norm.vmin) + norm.vmin for t in ticks]
        elif vmin > 0:
            ticks = [t * factor for t in ticks]
        else:
            ticks = [-t * factor for t in ticks][::-1]

    if horizontal_plot:
        cb = cbar.ColorbarBase(cax, cmap=cmap, norm=norm, ticks=ticks, orientation="horizontal",
                               ticklocation='top')
    else:
        cb = cbar.ColorbarBase(cax, cmap=cmap, norm=norm, ticks=ticks)
    if ticks_values_and_labels is not None:
        if horizontal_plot:
            cb.ax.set_xticklabels([str(t) for t in ticks_values_and_labels[1]])
        else:
            cb.ax.set_yticklabels([str(t) for t in ticks_values_and_labels[1]])
    if isinstance(label, str):
        cb.set_label(label, fontsize=fontsize)
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


def imshow_shifted(ax, param_name, values, visualization_extend, mask_2D=None):
    condition = np.isnan(values)
    if mask_2D is not None:
        condition |= mask_2D
    masked_array = np.ma.masked_where(condition, values)
    vmin, vmax = np.min(masked_array), np.max(masked_array)
    shifted_cmap = get_shifted_map(vmin, vmax)
    norm = get_norm(vmin, vmax)
    create_colorbase_axis(ax, param_name, shifted_cmap, norm)
    shifted_cmap.set_bad(color='white')
    if param_name != AbstractParams.SHAPE:
        epsilon = 1e-2 * (np.max(values) - np.min(values))
        value = np.min(values)
        # The right blue corner will be blue (but most of the time, another display will be on top)
        masked_array[-1, -1] = value - epsilon
    # IMPORTANT: Origin for all the plots is at the bottom left corner
    ax.imshow(masked_array, extent=visualization_extend, cmap=shifted_cmap, origin='lower')


def ticks_values_and_labels_for_percentages(graduation, max_abs_change):
    assert max_abs_change != np.inf
    positive_ticks = []
    tick = graduation
    while tick < max_abs_change:
        new_tick = round(tick, 2)
        positive_ticks.append(new_tick)
        tick += graduation
    all_ticks_labels = [-t for t in positive_ticks][::-1] + [0] + positive_ticks
    ticks_values = [((t / max_abs_change) + 1) / 2 for t in all_ticks_labels]
    return ticks_values, all_ticks_labels


def ticks_values_and_labels_for_half_value(graduation, max_abs_change, positive=True):
    assert max_abs_change != np.inf
    positive_ticks = []
    tick = 0
    while tick < max_abs_change:
        positive_ticks.append(round(tick, 1))
        tick += graduation
    ticks_values = [(t / max_abs_change) for t in positive_ticks]
    if not positive:
        positive_ticks = [-v for v in positive_ticks[::-1]]
        shift = 1 - ticks_values[-1]
        ticks_values = [t + shift for t in ticks_values]
    return ticks_values, positive_ticks


def ticks_values_and_labels_for_positive_value_with_min_abs_change(graduation, min_abs_change, max_abs_change):
    assert max_abs_change != np.inf
    positive_ticks = []
    tick = min_abs_change
    while tick < max_abs_change:
        positive_ticks.append(round(tick, 1))
        tick += graduation
    ticks_values = [((t - min_abs_change) / (max_abs_change - min_abs_change)) for t in positive_ticks]
    return ticks_values, positive_ticks


def ticks_and_labels_centered_on_one(max_ratio, min_ratio):
    """When we compute some ratio of two values.
    Then if we want to make a plot, when the color change from blue to red is at 1,
    then you should use this function"""
    # Option to have a number of graduation constant
    m = max(max_ratio / 1.0, 1.0 / min_ratio)
    max_ratio = 1.0 * m
    min_ratio = 1.0 / m
    # Build the middle point
    midpoint = (max_ratio - 1.0) / (max_ratio - 0)
    graduation = 0.1
    # Build lower graduation
    n = int(np.math.floor((1.0 - min_ratio) / graduation)) + 1
    a1 = midpoint / (1.0 - min_ratio)
    b1 = midpoint - 1.0 * a1
    xlist1 = [1.0 - i * graduation for i in range(n)]
    y_list1 = [a1 * x + b1 for x in xlist1]
    # Build upper graduation
    n = int(np.math.floor((max_ratio - 1.0) / graduation)) + 1
    xlist2 = [1.0 + i * graduation for i in range(n)]
    a2 = (1 - midpoint) / (max_ratio - 1.0)
    b2 = 1.0 - a2 * max_ratio
    y_list2 = [a2 * x + b2 for x in xlist2]
    labels = xlist1 + xlist2
    ticks = y_list1 + y_list2
    labels = [np.round(l, 1) for l in labels]
    return labels, max_ratio, midpoint, min_ratio, ticks


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


if __name__ == '__main__':
    cmap = plt.cm.bwr
    cmap = get_inverse_colormap(cmap)
    cmap = remove_the_extreme_colors(cmap)
    cmap = get_cmap_with_inverted_blue_and_green_channels(cmap)
