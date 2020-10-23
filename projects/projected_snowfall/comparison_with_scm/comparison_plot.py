import matplotlib.pyplot as plt
from typing import Dict

from matplotlib.lines import Line2D

from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_color, gcm_rcm_couple_to_str
from projects.projected_snowfall.comparison_with_scm.comparison_historical_visualizer import \
    ComparisonHistoricalVisualizer


def individual_plot(v):
    # v.adamont_studies.plot_maxima_time_series(v.massif_names, v.scm_study)
    v.shoe_plot_bias_maxima_comparison()
    # for plot_maxima in [True, False][:1]:
    #     v.plot_comparison(plot_maxima)


def collective_plot(altitude_to_visualizer):
    bias_of_the_mean_with_the_altitude(altitude_to_visualizer)


def bias_of_the_mean_with_the_altitude(altitude_to_visualizer: Dict[int, ComparisonHistoricalVisualizer]):
    visualizer = list(altitude_to_visualizer.values())[0]
    altitudes = list(altitude_to_visualizer.keys())
    for couple in visualizer.adamont_studies.gcm_rcm_couples:
        values = [v.mean_bias_maxima[couple] for v in altitude_to_visualizer.values()]

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
