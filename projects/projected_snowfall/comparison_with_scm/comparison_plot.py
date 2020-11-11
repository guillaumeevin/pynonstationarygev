import matplotlib.pyplot as plt
from typing import Dict

import numpy as np
from matplotlib.lines import Line2D

from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_color, gcm_rcm_couple_to_str, \
    scenario_to_str
from projects.projected_snowfall.comparison_with_scm.comparison_historical_visualizer import \
    ComparisonHistoricalVisualizer


def individual_plot(visualizer: ComparisonHistoricalVisualizer):
    # visualizer.adamont_studies.plot_maxima_time_series_adamont(visualizer.massif_names, visualizer.scm_study)
    # visualizer.shoe_plot_bias_maxima_comparison()
    for relative in [True, False]:
        visualizer.plot_map_with_the_mean_bias_in_the_mean(relative)

    visualizer.plot_map_with_the_rank()
    # for relative
    # for plot_maxima in [True, False][:1]:
    #     visualizer.plot_comparison(plot_maxima)


def collective_plot(altitude_to_visualizer: Dict[int, ComparisonHistoricalVisualizer]):
    visualizer = list(altitude_to_visualizer.values())[0]
    count_number_of_total_massifs = 0
    count_number_of_time_the_reanalysis_is_the_smallest = 0
    count_number_of_time_the_reanalysis_is_the_biggest = 0
    altitudes = list(altitude_to_visualizer.keys())
    all_ranks = []
    for v in altitude_to_visualizer.values():
        ranks = np.array(list(v.massif_name_to_rank.values()))
        count_number_of_total_massifs += len(ranks)
        count_number_of_time_the_reanalysis_is_the_smallest += sum(ranks == 0.0)
        all_ranks.extend(ranks)
    print(scenario_to_str(visualizer.study.scenario), visualizer.study.year_min, visualizer.study.year_max)
    print('Summary for rank for altitudes:', altitudes)
    print('Mean ranks:', np.mean(all_ranks))
    print('percentages of time reanalysis is the biggest:', 
          100 * count_number_of_time_the_reanalysis_is_the_biggest / count_number_of_total_massifs)
    print('number of time reanalysis is the biggest:', count_number_of_time_the_reanalysis_is_the_biggest,
          ' out of ', count_number_of_total_massifs, ' time series')
    print('percentages of time reanalysis is the smallest:', 
          100 * count_number_of_time_the_reanalysis_is_the_smallest / count_number_of_total_massifs)
    print('number of time reanalysis is the smallest:', count_number_of_time_the_reanalysis_is_the_smallest,
          ' out of ', count_number_of_total_massifs, ' time series')


def bias_of_the_mean_with_the_altitude(altitude_to_visualizer: Dict[int, ComparisonHistoricalVisualizer]):
    visualizer = list(altitude_to_visualizer.values())[0]
    altitudes = list(altitude_to_visualizer.keys())
    for couple in visualizer.adamont_studies.gcm_rcm_couples:
        values = [v.gcm_rcm_couple_to_bias_list_in_the_mean_maxima[couple] for v in altitude_to_visualizer.values()]

        ax = plt.gca()
        width = 10
        positions = [i * width * 2 for i in range(len(values))]
        bplot = ax.boxplot(values, positions=positions, widths=width, patch_artist=True, showmeans=True)
        color = gcm_rcm_couple_to_color[couple]
        for patch in bplot['boxes']:
            patch.set_facecolor(color)

        couple_name = ' + '.join(couple)
        plot_name = 'Mean bias w.r.t to the reanalysis for {}'.format(couple_name)
        ax.set_ylabel(plot_name)
        ax.set_xlabel('Altitude')
        ax.set_xticklabels(altitudes)
        ax.set_xlim([min(positions) - width, max(positions) + width])
        visualizer.plot_name = 'altitude_comparison/{}'.format(plot_name)
        visualizer.show_or_save_to_file(add_classic_title=False, no_title=True, tight_layout=True)
        ax.clear()
        plt.close()
