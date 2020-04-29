import matplotlib.pyplot as plt

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.create_shifted_cmap import \
    ticks_values_and_labels_for_percentages, get_shifted_map, get_colors


def load_plot(cmap, graduation, label, massif_name_to_value, altitude, fit_method):
    max_abs_change = max([abs(e) for e in massif_name_to_value.values()])
    ticks, labels = ticks_values_and_labels_for_percentages(graduation=graduation, max_abs_change=max_abs_change)
    min_ratio = -max_abs_change
    max_ratio = max_abs_change
    cmap = get_shifted_map(min_ratio, max_ratio, cmap)
    massif_name_to_color = {m: get_colors([v], cmap, min_ratio, max_ratio)[0]
                            for m, v in massif_name_to_value.items()}
    ticks_values_and_labels = ticks, labels
    ax = plt.gca()
    AbstractStudy.visualize_study(ax=ax,
                                  massif_name_to_value=massif_name_to_value,
                                  massif_name_to_color=massif_name_to_color,
                                  replace_blue_by_white=True,
                                  axis_off=False,
                                  cmap=cmap,
                                  show_label=False,
                                  add_colorbar=True,
                                  show=False,
                                  vmin=min_ratio,
                                  vmax=max_ratio,
                                  ticks_values_and_labels=ticks_values_and_labels,
                                  label=label,
                                  fontsize_label=10,
                                  )
    ax.get_xaxis().set_visible(True)
    ax.set_xticks([])
    ax.set_xlabel('Altitude = {}m'.format(altitude), fontsize=15)
    ax.set_title('Fit method is {}'.format(fit_method))