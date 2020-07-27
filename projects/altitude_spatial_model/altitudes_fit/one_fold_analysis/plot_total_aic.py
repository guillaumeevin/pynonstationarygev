import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_data.meteo_france_data.scm_models_data.visualization.utils import create_adjusted_axes
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.one_fold_fit import OneFoldFit
from projects.exceeding_snow_loads.utils import dpi_paper1_figure


def plots(visualizer):
    visualizer.plot_moments()
    visualizer.plot_best_coef_maps()
    visualizer.plot_shape_map()
    visualizer.plot_year_for_the_peak()


def plot_individual_aic(visualizer):
    OneFoldFit.best_estimator_minimizes_total_aic = False
    plots(visualizer)


def plot_total_aic(model_classes, visualizer):
    # Compute the mean AIC for each model_class
    OneFoldFit.best_estimator_minimizes_total_aic = True
    model_class_to_total_aic = {model_class: 0 for model_class in model_classes}
    model_class_to_name_str = {}
    # model_class_to_aic_scores = {model_class: [] for model_class in model_classes}
    # model_class_to_n = {model_class: [] for model_class in model_classes}
    for one_fold_fit in visualizer.massif_name_to_one_fold_fit.values():
        for model_class, estimator in one_fold_fit.model_class_to_estimator_with_finite_aic.items():
            model_class_to_total_aic[model_class] += estimator.aic()
            model_class_to_name_str[model_class] = estimator.margin_model.name_str
            # model_class_to_aic_scores[model_class].append(estimator.aic())
            # model_class_to_n[model_class].append(estimator.n())
    # model_class_to_mean_aic_score = {model_class: np.array(aic_scores).sum() / np.array(model_class_to_n[model_class]).sum()
    #                                  for model_class, aic_scores in model_class_to_aic_scores.items()}
    # print(model_class_to_mean_aic_score)
    sorted_model_class = sorted(model_classes, key=lambda m: model_class_to_total_aic[m])
    sorted_scores = [model_class_to_total_aic[model_class] for model_class in sorted_model_class]
    sorted_labels = [model_class_to_name_str[model_class] for model_class in sorted_model_class]
    print(sorted_model_class)
    print(sorted_scores)
    print(sorted_labels)
    best_model_class_for_total_aic = sorted_model_class[0]
    for one_fold_fit in visualizer.massif_name_to_one_fold_fit.values():
        one_fold_fit.best_estimator_class_for_total_aic = best_model_class_for_total_aic
    # Plot the ranking of the model based on their total aic
    plot_total_aic_repartition(visualizer, sorted_labels, sorted_scores)
    # Plot the results for the model that minimizes the mean aic
    plots(visualizer)


def plot_total_aic_repartition(visualizer, labels, scores):
    """
    Plot a single trend curves
    :return:
    """
    scores = np.array(scores)
    ax = create_adjusted_axes(1, 1)

    # parameters
    width = 3
    size = 30
    legend_fontsize = 30
    labelsize = 10
    linewidth = 3
    x = [2 * width * (i + 1) for i in range(len(labels))]
    ax.bar(x, scores, width=width, color='grey', edgecolor='grey', label='Total Aic',
           linewidth=linewidth)
    ax.legend(loc='upper right', prop={'size': size})
    ax.set_ylabel(' Total AIC score \n '
                  'i.e. sum of AIC score for all massifs ', fontsize=legend_fontsize)
    ax.set_xlabel('Models', fontsize=legend_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.set_xticks(x)
    ax.set_ylim([scores.min(), scores.max()])
    ax.yaxis.grid()
    ax.set_xticklabels(labels)

    # Save plot
    visualizer.plot_name = 'Total AIC ranking'
    visualizer.show_or_save_to_file(no_title=True, dpi=dpi_paper1_figure)
    plt.close()
