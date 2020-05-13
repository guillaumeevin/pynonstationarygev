from typing import Dict
import matplotlib.pyplot as plt

from extreme_data.meteo_france_data.scm_models_data.visualization.utils import create_adjusted_axes
from projects.exceeding_snow_loads.utils import dpi_paper1_figure
from extreme_trend.visualizers.study_visualizer_for_non_stationary_trends import StudyVisualizerForNonStationaryTrends


def permute(l, permutation):
    # permutation = [i//2  if i % 2 == 0 else 4 + i //2 for i in range(8)]
    return [l[i] for i in permutation]

def plot_selection_curves(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends]):
    """
    Plot a single trend curves
    :return:
    """
    visualizer = list(altitude_to_visualizer.values())[0]

    ax = create_adjusted_axes(1, 1)

    selected_counter = merge_counter([v.selected_trend_test_class_counter for v in altitude_to_visualizer.values()])
    selected_and_significative_counter = merge_counter([v.selected_and_significative_trend_test_class_counter for v in altitude_to_visualizer.values()])
    # selected_and_significative_counter = merge_counter([v.selected_and_anderson_goodness_of_fit_trend_test_class_counter for v in altitude_to_visualizer.values()])
    # selected_and_significative_counter = merge_counter([v.selected_and_kstest_goodness_of_fit_trend_test_class_counter for v in altitude_to_visualizer.values()])
    total_of_selected_models = sum(selected_counter.values())
    l = sorted(enumerate(visualizer.non_stationary_trend_test), key=lambda e: selected_counter[e[1]])
    permutation = [i for i, v in l][::-1]

    select_list = get_ordered_list_from_counter(selected_counter, total_of_selected_models, visualizer, permutation)
    selected_and_signifcative_list = get_ordered_list_from_counter(selected_and_significative_counter, total_of_selected_models, visualizer, permutation)
    labels = permute(['${}$'.format(t.label) for t in visualizer.non_stationary_trend_test], permutation)

    print(l)
    print(select_list)
    print(selected_and_signifcative_list)
    # [(5, <    class 'data.trend_analysis.univariate_test.extreme_trend.trend_test_two_parameters.gev_trend_test_two_parameters.GevLocationAgainstGumbel'> ), (6, < class 'data.trend_analysis.univariate_test.extreme_trend.trend_test_two_parameters.gev_trend_test_two_parameters.GevScaleAgainstGumbel' > ), (2, < class 'data.trend_analysis.univariate_test.extreme_trend.trend_test_one_parameter.gumbel_trend_test_one_parameter.GumbelScaleTrendTest' > ), (1, < class 'data.trend_analysis.univariate_test.extreme_trend.trend_test_one_parameter.gumbel_trend_test_one_parameter.GumbelLocationTrendTest' > ), (7, < class 'data.trend_analysis.univariate_test.extreme_trend.trend_test_three_parameters.gev_trend_test_three_parameters.GevLocationAndScaleTrendTestAgainstGumbel' > ), (3, < class 'data.trend_analysis.univariate_test.extreme_trend.trend_test_two_parameters.gumbel_test_two_parameters.GumbelLocationAndScaleTrendTest' > ), (4, < class 'data.trend_analysis.univariate_test.extreme_trend.trend_test_one_parameter.gumbel_trend_test_one_parameter.GevStationaryVersusGumbel' > ), (0, < class 'data.trend_analysis.univariate_test.extreme_trend.trend_test_one_parameter.gumbel_trend_test_one_parameter.GumbelVersusGumbel' > )]
    # [32.64462809917355, 24.380165289256198, 12.396694214876034, 9.50413223140496, 9.090909090909092, 5.785123966942149, 3.71900826446281, 2.479338842975207]
    # [0, 13.223140495867769, 7.851239669421488, 8.264462809917354, 4.958677685950414, 2.479338842975207, 0.8264462809917356, 2.0661157024793386]

    # parameters
    width = 5
    size = 30
    legend_fontsize = 30
    labelsize = 25
    linewidth = 3
    x = [10 * (i+1) for i in range(len(select_list))]
    ax.bar(x, select_list, width=width, color='grey', edgecolor='grey', label='Non significant model',
           linewidth=linewidth)
    ax.bar(x, selected_and_signifcative_list, width=width, color='black', edgecolor='black',
           label='Significant model',
           linewidth=linewidth)
    ax.legend(loc='upper right', prop={'size': size})
    ax.set_ylabel(' Frequency of selected model w.r.t all time series \n '
                  'i.e. for all massifs and altitudes (\%)', fontsize=legend_fontsize)
    ax.set_xlabel('Models', fontsize=legend_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.set_xticks(x)
    ax.yaxis.grid()
    ax.set_xticklabels(labels)

    # for ax_horizontal in [ax, ax_twiny]:
    #     if ax_horizontal == ax_twiny:
    #         ax_horizontal.plot(altitudes, [0 for _ in altitudes], linewidth=0)
    #     else:
    #         ax_horizontal.set_xlabel('Altitude', fontsize=legend_fontsize)
    #     ax_horizontal.set_xticks(altitudes)
    #     ax_horizontal.set_xlim([700, 5000])
    #     ax_horizontal.tick_params(labelsize=labelsize)
    #
    # # Set the number of massifs on the upper axis
    # ax_twiny.set_xticklabels([v.study.nb_study_massif_names for v in altitude_to_visualizer.values()])
    # ax_twiny.set_xlabel('Total number of massifs at each altitude (for the percentage)', fontsize=legend_fontsize)
    #
    # ax.set_ylabel('Massifs with decreasing trend (\%)', fontsize=legend_fontsize)
    # max_percent = int(max(percent_decrease))
    # n = 2 + (max_percent // 10)
    # ax_ticks = [10 * i for i in range(n)]
    # # upper_lim = max_percent + 3
    # upper_lim = n + 5
    # ax_lim = [0, upper_lim]
    # for axis in [ax, ax_twinx]:
    #     axis.set_ylim(ax_lim)
    #     axis.set_yticks(ax_ticks)
    #     axis.tick_params(labelsize=labelsize)
    # ax.yaxis.grid()
    #
    # label_curve = (visualizer.label).replace('change', 'decrease')
    # ax_twinx.set_ylabel(label_curve.replace('', ''), fontsize=legend_fontsize)
    # for region_name, mean_decrease in zip(AbstractExtendedStudy.region_names, mean_decreases):
    #     if len(mean_decreases) > 1:
    #         label = region_name
    #     else:
    #         label = 'Mean relative decrease'
    #     ax_twinx.plot(altitudes, mean_decrease, label=label, linewidth=linewidth, marker='o')
    #     ax_twinx.legend(loc='upper right', prop={'size': size})

    # Save plot
    visualizer.plot_name = 'Selection curves'
    visualizer.show_or_save_to_file(no_title=True, dpi=dpi_paper1_figure)
    plt.close()


def get_ordered_list_from_counter(selected_counter, total_of_selected_models, visualizer, permutation):
    return permute([100 * float(selected_counter[t]) / total_of_selected_models if t in selected_counter else 0
                for t in visualizer.non_stationary_trend_test], permutation)

def merge_counter(counters_list):
    global_counter = counters_list[0]
    for c in counters_list[1:]:
        global_counter += c
    return global_counter

