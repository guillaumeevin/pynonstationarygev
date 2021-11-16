import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.utils import r
from root_utils import VERSION_TIME


def plot_density_vertically(ax, color, gev_params: GevParams, t, ylim_upper_for_plots, linewidth,
                            normalize_factor=None, scaling_of_density=1.0, label=None):
    yp = np.linspace(-4, ylim_upper_for_plots, num=1000)
    xp = gev_params.density(yp)
    if normalize_factor is None:
        return xp.max()
    else:
        xp = xp * scaling_of_density / normalize_factor
        xp = t + xp
        l = [(xpp, ypp) for xpp, ypp in zip(xp, yp) if ypp <= ylim_upper_for_plots]
        xp, yp = zip(*l)
        ax.plot(xp, yp, color=color, linestyle='-', linewidth=linewidth, label=label)


def load_gev_params(shape, t, loc_plot=True):
    loc, scale, shape = 1, 1, shape
    if loc_plot:
        gev_params = GevParams(loc + t, scale, shape)
    else:
        gev_params = GevParams(loc, scale + t, shape)
    return gev_params



def non_stationary_gev_plot(loc_plot=True):
    ax = plt.gca()
    ax2 = ax.twinx()

    normalization_factor = 0
    t_for_density = [0, 1, 2]

    right_limit = 3
    left_limit = -0.2
    xtick = 0

    scaling_of_density = 0.75

    length_x = right_limit - left_limit

    linewidth_big_plot = 4
    ylim_upper_for_plots = 8.5
    percentage_upper_for_plots = 0.735
    ylim_upper = (1 / percentage_upper_for_plots) * ylim_upper_for_plots
    colors = ['tab:blue', 'tab:red', 'tab:green']

    # Compute max scale
    max_scale = 0
    linewidth = 4
    for color, shape in zip(colors, [-1, 0, 1]):
        for t in t_for_density:
            gev_params = load_gev_params(shape, t, loc_plot)
            res = plot_density_vertically(ax, color, gev_params, t, ylim_upper_for_plots,
                                          linewidth, None, scaling_of_density)
            max_scale = max(max_scale, res)
    print(max_scale)
    # Plot density for real
    for color, shape in zip(colors, [-1, 0, 1]):
        for j, t in enumerate(t_for_density):
            gev_params = load_gev_params(shape, t, loc_plot)
            if j > 0:
                label = None
            elif loc_plot:
                label= "$\\mu(t) = 1 + t, \\sigma(t) = 1, \\xi(t) = {}$".format(shape)
            else:
                label= "$\\mu(t) = 1, \\sigma(t) = 1 + t, \\xi(t) = {}$".format(shape)
                # label = '\mu = 1, \sigma = 1 + t, \xi = {}'.format(shape)
            plot_density_vertically(ax, color, gev_params, t, ylim_upper_for_plots,
                                          linewidth, max_scale, scaling_of_density,label)

    for t in t_for_density[:]:
        c = np.array([t])
        scale_for_upper_arrow = 0.05
        text_height = -1.3
        ax.axhline(text_height, (t + scaling_of_density - left_limit) / length_x,
                   (t - left_limit) / length_x, color='k', linewidth=1)
        ax.axvline(t, 0, percentage_upper_for_plots + scale_for_upper_arrow, color='k', linewidth=1,
                   linestyle='dotted')
        epsilon = 0.02
        start_text = t + 0.05
        # Draw arrow
        size_arrow = 0.03
        ylims = ax.get_ylim()
        xlims = ax.get_xlim()
        scaling_for_y = (ylims[1] - ylims[0]) / (xlims[1] - xlims[0]) / percentage_upper_for_plots
        start_arrow = t + scaling_of_density
        # horizontal arrow
        ax.plot([start_arrow, start_arrow - size_arrow],
                [text_height, text_height + size_arrow * scaling_for_y], color='k')
        ax.plot([start_arrow, start_arrow - size_arrow],
                [text_height, text_height - size_arrow * scaling_for_y], color='k')
        # vertical arrow
        scaling = 1.1
        ylim_upper_arrow = ylim_upper * (percentage_upper_for_plots + scale_for_upper_arrow)
        # ax.plot([t, t + size_arrow], [ylim_upper_arrow, ylim_upper_arrow - size_arrow * scaling_for_y * scaling], color='k')
        # ax.plot([t, t - size_arrow], [ylim_upper_arrow, ylim_upper_arrow - size_arrow * scaling_for_y * scaling], color='k')
        # for i in range(5):
        #     tick_x = start_text + 0.05 + 0.05 * i
        #     ax.axvline(tick_x, percentage_upper_for_plots - epsilon,
        #                percentage_upper_for_plots, color='k', linewidth=1)
        ax.text(start_text, text_height + 0.2, "Probability density function",
                fontsize=6)

    for color, shape in zip(colors, [-1, 0, 1]):
        t_list = np.linspace(left_limit, right_limit, 100)
        y_list = []
        for t in t_list:
            gev_params = load_gev_params(shape, t, loc_plot)
            y = gev_params.return_level(4)
            y_list.append(y)
        ax.plot(t_list, y_list, color=color, linewidth=linewidth, linestyle='--')

    # Final plt
    ylabel = 'y, an annual maxima'
    ax.set_ylabel(ylabel)
    xlabel = 't, a temporal covariate'
    ax.set_xlabel(xlabel)
    ax.set_xlim((left_limit, right_limit))
    xticks = []
    while xtick <= right_limit:
        xticks.append(xtick)
        xtick += 1
    ax.set_xticks(xticks)
    ax.set_xticklabels(['{}'.format(int(h) if int(h) == h else h) for h in xticks])

    ylim = (-2, ylim_upper)
    ax.set_ylim(ylim)
    ax2.set_ylim(ylim)
    ax2.set_yticks([])
    handles, labels = ax.get_legend_handles_labels()
    handles[:2] = handles[:2][::-1]
    labels[:2] = labels[:2][::-1]
    legendsize = 8.5
    loc = "upper left"
    ax.legend(handles, labels, prop={'size': legendsize}, loc=loc,
                    handlelength=4)

    # Build legend for the different style
    labels = ["Probability density function",
              "4-year return level"]
    handles = [
        Line2D([0], [0], color='k', linestyle=linestyle)
        for linestyle in ["-", "--"]
    ]
    leg = ax2.legend(handles=handles, labels=labels, loc='upper right', prop={'size': legendsize},
                     handlelength=3)

    # plt.show()
    # visualizer.plot_name = title
    # visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)
    # plt.close()
    filename = "{}/{}".format(VERSION_TIME, "non stationary plot {}".format(loc_plot))
    StudyVisualizer.savefig_in_results(filename, transparent=True)
    plt.close()


if __name__ == '__main__':
    # non_stationary_gev_plot(True)
    for loc_plot in [True, False]:
        non_stationary_gev_plot(loc_plot)
