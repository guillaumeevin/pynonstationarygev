import matplotlib.pyplot as plt

from projects.exceeding_snow_loads.paper_utils import ModelSubsetForUncertainty, dpi_paper1_figure
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes


def plot_diagnosis_risk(altitude_to_visualizer):
    ax = plt.gca()
    altitudes = list(altitude_to_visualizer.keys())
    visualizers = list(altitude_to_visualizer.values())
    visualizer = visualizers[0]
    model_subset_for_uncertainty = ModelSubsetForUncertainty.non_stationary_gumbel_and_gev
    ci_method = ConfidenceIntervalMethodFromExtremes.ci_mle

    plot_mean_exceedance(visualizers, altitudes, ax, model_subset_for_uncertainty, ci_method)

    visualizer.plot_name = 'Diagnosis plot'
    visualizer.show_or_save_to_file(no_title=True, dpi=dpi_paper1_figure)
    plt.close()

    plt.show()


def plot_mean_exceedance(visualizers, altitudes, ax, model_subset_for_uncertainty, ci_method):
    l = [v.excess_metrics(ci_method, model_subset_for_uncertainty)[3:] for v in visualizers]
    diff, diff_c, diff_e = zip(*l)
    ax.plot(altitudes, diff, marker='o', color='red', label='diff')
    ax.plot(altitudes, diff_c, marker='o', color='yellow', label='diff c')
    ax.plot(altitudes, diff_e, marker='o', color='orange', label='diff e')

    ax.set_ylabel("Mean exceedance (kN)", fontsize=15)
    ax.set_xlabel("Altitude", fontsize=15)
    ax.yaxis.grid()

    ax.legend()
