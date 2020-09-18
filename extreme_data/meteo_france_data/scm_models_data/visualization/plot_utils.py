import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.create_shifted_cmap import \
    ticks_values_and_labels_for_percentages, get_shifted_map, get_colors, ticks_values_and_labels_for_positive_value, \
    get_half_colormap


def plot_against_altitude(x_ticks, ax, massif_id, massif_name, values, altitude=None, fill=False, massif_name_as_labels=True):
    if massif_name_as_labels:
        di = massif_id // 8
        if di == 0:
            linestyle = '-'
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
    else:
        color = None
        linestyle = None
        label = '{} m'.format(altitude)
    if not fill:
        ax.plot(x_ticks, values, color=color, linewidth=2, label=label, linestyle=linestyle)
    else:
        lower_bound, upper_bound = zip(*values)
        # ax.fill_between(altitudes, lower_bound, upper_bound, color=color, alpha=0.2, label=label + '95\% confidence interval')
        ax.fill_between(x_ticks, lower_bound, upper_bound, color=color, alpha=0.2)


def load_plot(cmap, graduation, label, massif_name_to_value, altitude, fit_method, add_x_label=True,
              negative_and_positive_values=True, massif_name_to_text=None, add_colorbar=True, max_abs_change=None):
    if max_abs_change is None:
        max_abs_change = max([abs(e) for e in massif_name_to_value.values()])
    if negative_and_positive_values:
        ticks, labels = ticks_values_and_labels_for_percentages(graduation=graduation, max_abs_change=max_abs_change)
        min_ratio = -max_abs_change
        max_ratio = max_abs_change
        cmap = get_shifted_map(min_ratio, max_ratio, cmap)
    else:
        ticks, labels = ticks_values_and_labels_for_positive_value(graduation=graduation, max_abs_change=max_abs_change)
        cmap = get_half_colormap(cmap)
        min_ratio = 0
        max_ratio = max_abs_change

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
                                  fontsize_label=10,
                                  massif_name_to_text=massif_name_to_text,
                                  add_text=massif_name_to_text is not None,
                                  massif_name_to_hatch_boolean_list=massif_name_to_hatch_boolean_list
                                  )
    ax.get_xaxis().set_visible(True)
    ax.set_xticks([])
    if add_x_label:
        ax.set_xlabel('Altitude = {}m'.format(altitude), fontsize=15)
        # ax.set_title('Fit method is {}'.format(fit_method))
