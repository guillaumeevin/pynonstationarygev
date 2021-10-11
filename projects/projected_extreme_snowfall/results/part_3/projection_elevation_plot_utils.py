import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit


def drawPieMarker(ax, xs, ys, probabilities, sizes, colors):
    assert sum(probabilities) == 1, 'sum of ratios needs to be = 1'

    markers = []
    previous = 0
    # calculate the points of the pie pieces
    for color, probability in zip(colors, probabilities):
        this = 2 * np.pi * probability + previous
        x = [0] + np.cos(np.linspace(previous, this, 10)).tolist() + [0]
        y = [0] + np.sin(np.linspace(previous, this, 10)).tolist() + [0]
        xy = np.column_stack([x, y])
        previous = this
        markers.append({'marker': xy, 's': np.abs(xy).max() ** 2 * np.array(sizes), 'facecolor': color})

    # scatter each of the pie pieces to create pies
    for marker in markers:
        ax.scatter(xs, ys, **marker)


def plot_pychart_scatter_plot(visualizer_list, massif_names, covariates, with_return_level):
    ax = plt.gca()
    sizes = [1500 for _ in range(2)]
    altitudes = []
    for visualizer in visualizer_list:
        altitude = visualizer.study.altitude
        altitudes.append(altitude)
        for covariate in covariates:
            OneFoldFit.COVARIATE_AFTER_TEMPERATURE = covariate
            _, *all_l = visualizer.all_trends(massif_names, with_significance=False,
                                              with_return_level=with_return_level)
            decreasing_percentage = all_l[0] / 100
            if decreasing_percentage == 1:
                probabilities = [0.25 for _ in range(4)]
                colors = ['blue' for _ in range(4)]
            elif decreasing_percentage == 0:
                probabilities = [0.25 for _ in range(4)]
                colors = ['red' for _ in range(4)]
            else:
                probabilities = [decreasing_percentage / 2, decreasing_percentage / 2,
                                 (1 - decreasing_percentage) / 2, (1 - decreasing_percentage) / 2]
                colors = ['blue', 'blue', 'red', 'red']
            drawPieMarker(ax, covariate, altitude, probabilities, sizes, colors)

    legend_fontsize = 10
    labelsize = 10
    ax.set_xlabel('T, the smoothed anomaly of global temperature w.r.t. pre-industrial levels (K)', fontsize=legend_fontsize)
    ax.set_ylabel('Elevation (m)', fontsize=legend_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.set_xticks(covariates)
    ax.set_yticks(altitudes)
    mi, ma = ax.get_ylim()
    border = 200
    ax.set_ylim((mi - border, ma + border + 450))
    ax.set_yticklabels(["{} m".format(v.study.altitude) for v in visualizer_list])
    ax.set_xticklabels(["+{}".format(int(c) if int(c) == c else c) for c in covariates])

    # Build legend
    custom_lines = [Line2D([0], [0], color=color, lw=6) for color in ['blue', 'red']]
    # custom_lines = [Patch(facecolor=color, edgecolor=color) for color in ['blue', 'red']]
    labels = ['\% of massifs with {} between +T and +1 degrees'.format(s) for s in ['a decrease', 'an increase']]
    ax.legend(custom_lines, labels, prop={'size': 12}, loc='upper left' )

    label = "return level" if with_return_level else "mean"
    visualizer = visualizer_list[0]
    visualizer.plot_name = 'Piece charts for {}'.format(label)
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

    plt.close()
