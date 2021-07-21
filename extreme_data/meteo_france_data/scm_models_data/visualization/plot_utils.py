import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.create_shifted_cmap import \
    ticks_values_and_labels_for_percentages, get_shifted_map, get_colors, ticks_values_and_labels_for_half_value, \
    get_upper_half_colormap, ticks_values_and_labels_for_positive_value_with_min_abs_change, get_lower_half_colormap


def plot_against_altitude(x_ticks, ax, massif_id, massif_name, values, altitude=None, fill=False,
                          massif_name_as_labels=True,
                          elevation_as_xaxis=True, legend=False):
    if massif_name_as_labels:
        color, linestyle, label = get_color_and_linestyle_from_massif_id(massif_id, massif_name)
    else:
        color = None
        linestyle = None
        label = '{} m'.format(altitude)
    if not fill:
        args = [x_ticks, values] if elevation_as_xaxis else [values, x_ticks]
        if legend:
            ax.plot(*args, color=color, linewidth=2, label=label, linestyle=linestyle)
        else:
            ax.plot(*args, color=color, linewidth=2, label=label, linestyle=linestyle, marker='o')
    else:
        assert elevation_as_xaxis, NotImplementedError('todo')
        lower_bound, upper_bound = zip(*values)
        # ax.fill_between(altitudes, lower_bound, upper_bound, color=color, alpha=0.2, label=label + '95\% confidence interval')
        ax.fill_between(x_ticks, lower_bound, upper_bound, color=color, alpha=0.2)


def get_color_and_linestyle_from_massif_id(massif_id, massif_name):
    di = massif_id // 8
    if di == 0:
        linestyle = '-.'
    elif di == 1:
        linestyle = 'dotted'
    else:
        linestyle = '--'
    colors = list(mcolors.TABLEAU_COLORS)
    colors[-3:-1] = []  # remove gray and olive
    color = colors[massif_id % 8]
    # Label
    massif_name_str = ' '.join(massif_name.split('_'))
    label = massif_name_str
    return color, linestyle, label


def load_plot(cmap, graduation, label, massif_name_to_value, altitude, add_x_label=True,
              negative_and_positive_values=True, massif_name_to_text=None, add_colorbar=True, max_abs_change=None,
              xlabel=None, fontsize_label=10, massif_names_with_white_dot=None,
              min_ratio_equal_to_zero_for_positive_values=True,
              half_cmap_for_positive=True,
              ):
    if max_abs_change is None:
        max_abs_change = max([abs(e) for e in massif_name_to_value.values()])
    if negative_and_positive_values:
        ticks, labels = ticks_values_and_labels_for_percentages(graduation=graduation, max_abs_change=max_abs_change)
        min_ratio = -max_abs_change
        max_ratio = max_abs_change
        cmap = get_shifted_map(min_ratio, max_ratio, cmap)
    else:
        max_ratio = max_abs_change
        is_negative = all([e < 0 for e in massif_name_to_value.values()])
        if half_cmap_for_positive:
            if is_negative:
                cmap = get_lower_half_colormap(cmap)
            else:
                cmap = get_upper_half_colormap(cmap)
        if min_ratio_equal_to_zero_for_positive_values:
            ticks, labels = ticks_values_and_labels_for_half_value(graduation=graduation,
                                                                   max_abs_change=max_abs_change,
                                                                   positive=not is_negative)
            min_ratio = 0
        else:
            min_ratio = np.floor(min([abs(e) for e in massif_name_to_value.values()]))
            ticks, labels = ticks_values_and_labels_for_positive_value_with_min_abs_change(graduation=graduation,
                                                                                           min_abs_change=min_ratio,
                                                                                           max_abs_change=max_ratio)
        if is_negative:
            min_ratio, max_ratio = -max_ratio, -min_ratio

    for v in massif_name_to_value.values():
        assert isinstance(v, float)
    massif_name_to_color = {m: get_colors([v], cmap, min_ratio, max_ratio)[0]
                            for m, v in massif_name_to_value.items()}

    ticks_values_and_labels = ticks, labels
    ax = plt.gca()

    massif_name_to_hatch_boolean_list = {}
    for massif_name in set(AbstractStudy.all_massif_names()) - set(list(massif_name_to_value.keys())):
        massif_name_to_hatch_boolean_list[massif_name] = [True, True]

    AbstractStudy.visualize_study(ax=ax,
                                  massif_name_to_value=massif_name_to_value,
                                  massif_name_to_color=massif_name_to_color,
                                  replace_blue_by_white=True,
                                  axis_off=False,
                                  cmap=cmap,
                                  show_label=False,
                                  add_colorbar=add_colorbar,
                                  show=False,
                                  vmin=min_ratio,
                                  vmax=max_ratio,
                                  ticks_values_and_labels=ticks_values_and_labels,
                                  label=label,
                                  fontsize_label=fontsize_label,
                                  massif_name_to_text=massif_name_to_text,
                                  add_text=massif_name_to_text is not None,
                                  massif_name_to_hatch_boolean_list=massif_name_to_hatch_boolean_list,
                                  massif_names_with_white_dot=massif_names_with_white_dot
                                  )
    ax.get_xaxis().set_visible(True)
    ax.set_xticks([])
    if add_x_label:
        if xlabel is None:
            ax.set_xlabel('Altitude = {}m'.format(altitude), fontsize=15)
        else:
            ax.set_xlabel(xlabel, fontsize=10)
        # ax.set_title('Fit method is {}'.format(fit_method))
