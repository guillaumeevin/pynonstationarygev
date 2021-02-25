from typing import List
import matplotlib.pyplot as plt
import numpy as np

from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from extreme_trend.one_fold_analysis.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from extreme_trend.one_fold_analysis.one_fold_fit import OneFoldFit


def plot_coherence_curves(massif_names, visualizer_list: List[AltitudesStudiesVisualizerForNonStationaryModels]):
    folder = 'Coherence'
    elevation_as_xaxis = False
    visualizer = visualizer_list[0]
    all_valid_names = set.union(*[v.get_valid_names(massif_names) for v in visualizer_list])
    for massif_name in all_valid_names:

        # For plotting the legend
        legend = False
        colors = ['blue', 'red', 'green']
        elevational_str = 'Piecewise elevational-temporal models in'
        labels = ['{} 1959'.format(elevational_str), '{} 2019'.format(elevational_str), 'Pointwise distributions']
        altitudinal_model = [True, True, False]
        years = [1959, 2019, None]
        for i in list(range(4))[:]:
            # Load ax
            ax = plt.gca()
            if i % 2 == 1:
                ax.set_yticks([])
                ax2 = ax.twinx()
            else:
                ax2 = ax
            if not elevation_as_xaxis and i < 2:
                ax2.set_xticks([])
                ax3 = ax2.twiny()
            else:
                ax3 = ax2

            for color, global_label, is_altitudinal, year in list(zip(colors, labels, altitudinal_model, years))[:]:
                x_all_list, values_all_list, labels, all_bound_list = load_all_list(massif_name, visualizer_list,
                                                                                    is_altitudinal,
                                                                                    year)
                label = labels[i]
                label = label.capitalize()

                # Set labels
                fontsize_label = 15

                for a in [ax, ax2, ax3]:
                    a.tick_params(labelsize=fontsize_label)

                if elevation_as_xaxis:
                    ax3.set_xlabel('Elevation (m)', fontsize=fontsize_label)
                    ax2.set_ylabel(label, fontsize=fontsize_label)
                else:
                    ax2.set_ylabel('Elevation (m)', fontsize=fontsize_label)
                    if i == 3:
                        ax.set_xlabel(label, fontsize=fontsize_label)
                    else:
                        ax3.set_xlabel(label, fontsize=fontsize_label)

                plot_coherence_curve(ax3, i, x_all_list, values_all_list, all_bound_list,
                                     is_altitudinal, color, global_label, year, legend,
                                     elevation_as_xaxis)
            visualizer.plot_name = '{}/{}_{}'.format(folder, massif_name.replace('_', '-'), label)
            visualizer.show_or_save_to_file(add_classic_title=False, no_title=True, dpi=200)
            plt.close()


def plot_coherence_curve(ax, i, x_all_list, values_all_list, all_bound_list,
                         is_altitudinal, color, global_label, year, legend, elevation_as_xaxis):
    legend_line = False
    # Plot with complete line
    for j, (x_list, value_list) in enumerate(list(zip(x_all_list, values_all_list))):
        value_list_i = value_list[i]
        label_plot = global_label if j == 0 else None
        args = [x_list, value_list_i] if elevation_as_xaxis else [value_list_i, x_list]

        if is_altitudinal:
            if legend and legend_line:
                ax.plot(*args, linestyle='solid', color=color, label=label_plot, linewidth=5)
            else:
                ax.plot(*args, linestyle='solid', color=color)
        else:
            if legend and legend_line:
                ax.plot(*args, linestyle='None', color=color, label=label_plot, marker='o', markersize=10)
            else:
                ax.plot(*args, linestyle='None', color=color, marker='o')
                ax.plot(*args, linestyle='dotted', color=color)

    # Plot with dotted line
    for x_list_before, value_list_before, x_list_after, value_list_after in zip(x_all_list, values_all_list,
                                                                                x_all_list[1:],
                                                                                values_all_list[1:]):
        x_list = [x_list_before[-1], x_list_after[0]]
        value_list_dotted = [value_list_before[i][-1], value_list_after[i][0]]
        args = [x_list, value_list_dotted] if elevation_as_xaxis else [value_list_dotted, x_list]
        ax.plot(*args, linestyle='dotted', color=color)

    # Plot confidence interval
    if i == 3 and year in [None, 2019]:
        for j, (x_list, bounds) in enumerate(list(zip(x_all_list, all_bound_list))):
            if len(bounds) > 0:
                lower_bound, upper_bound = bounds
                f = ax.fill_between if elevation_as_xaxis else ax.fill_betweenx
                if legend and not legend_line:
                    model_name = 'piecewise elevational-temporal models in 2019' if is_altitudinal else 'pointwise distributions'
                    percentage = AbstractExtractEurocodeReturnLevel.percentage_confidence_interval
                    fill_label = "{}\% confidence interval for the {}".format(percentage, model_name) if j == 0 else None
                    f(x_list, lower_bound, upper_bound, color=color, alpha=0.2, label=fill_label)
                else:
                    f(x_list, lower_bound, upper_bound, color=color, alpha=0.2)

    if legend:
        print("here")
        min, max = ax.get_ylim()
        ax.set_ylim([min, 2 * max])
        size = 15 if legend_line else 11
        ax.legend(prop={'size': size})


def load_all_list(massif_name, visualizer_list, altitudinal_model=True, year=2019):
    all_altitudes_list = []
    all_values_list = []
    all_bound_list = []
    for visualizer in visualizer_list:

        if massif_name in visualizer.massif_name_to_one_fold_fit:
            if altitudinal_model:
                # Piecewise altitudinal temporal model
                min_altitude, *_, max_altitude = visualizer.massif_name_to_massif_altitudes[massif_name]
                one_fold_fit = visualizer.massif_name_to_one_fold_fit[massif_name]
                step = 300
                altitudes_list = list(range(min_altitude, max_altitude + step, step))
                gev_params_list = [one_fold_fit.get_gev_params(altitude, year) for altitude in altitudes_list]
                confidence_interval_values = [one_fold_fit.best_confidence_interval(altitude, year) for altitude in
                                              altitudes_list]
            else:
                # Pointwise distribution
                assert OneFoldFit.return_period == 100, 'change the call below'
                altitudes_list, study_list_valid = zip(*[(a, s) for a, s in visualizer.studies.altitude_to_study.items()
                                                         if massif_name in
                                                         s.massif_name_to_stationary_gev_params_and_confidence
                                                             (OneFoldFit.quantile_level,
                                                             visualizer.confidence_interval_based_on_delta_method)[
                                                             0]])
                gev_params_list = [
                    study.massif_name_to_stationary_gev_params_and_confidence
                                                             (OneFoldFit.quantile_level,
                                                             visualizer.confidence_interval_based_on_delta_method)[0][massif_name]
                    for study in study_list_valid]
                confidence_interval_values = [
                    study.massif_name_to_stationary_gev_params_and_confidence
                                                             (OneFoldFit.quantile_level,
                                                             visualizer.confidence_interval_based_on_delta_method)[1][massif_name]
                    for study in study_list_valid]

            # Checks
            values = [(gev_params.location, gev_params.scale, gev_params.shape,
                       gev_params.return_level(return_period=OneFoldFit.return_period))
                      for gev_params in gev_params_list]
            for a, b in zip(values, confidence_interval_values):
                if not np.isnan(b.mean_estimate):
                    assert np.isclose(a[-1], b.mean_estimate)
            bound_list = [c.confidence_interval for c in confidence_interval_values if not np.isnan(c.mean_estimate)]
            values_list = list(zip(*values))
            all_values_list.append(values_list)
            all_altitudes_list.append(altitudes_list)
            all_bound_list.append(list(zip(*bound_list)))
    labels = ['location', 'scale', 'shape', '{}-year return levels'.format(OneFoldFit.return_period)]

    for i, label in enumerate(labels):
        if i < 3:
            label += ' parameter'
        unit = 'no unit' if i == 2 else visualizer_list[0].study.variable_unit
        label += ' ({})'.format(unit)
        labels[i] = label

    return all_altitudes_list, all_values_list, labels, all_bound_list
