import numpy as np
import matplotlib.pyplot as plt

from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_fit.model.utils import r
from root_utils import VERSION_TIME


def gev_plot():
    lim = 5
    x = np.linspace(-lim, lim, 100)
    loc, scale = 1, 1
    shapes = [-1, 0, 1]
    for shape in shapes:
        label = '$\zeta= {} $'.format(shape)
        y = r.dgev(x, loc, scale, shape)
        plt.plot(x, y, label=label)
    plt.legend()
    plt.xlabel('$y$')
    plt.ylabel('$f_{GEV}(y|1,1,\zeta)$')
    plt.show()


def gev_plot_big():
    ax = plt.gca()
    lim = 5
    x = np.linspace(-lim, lim, 100)
    loc, scale = 1, 1
    shapes = [-1, 0, 1]
    colors = ['tab:blue', 'tab:red', 'tab:green']
    for shape, color in zip(shapes, colors):
        label = '$\zeta= {} $'.format(shape)
        y = r.dgev(x, loc, scale, shape)
        ax.plot(x, y, label=label, linewidth=5, color=color)
    ax.set_ylim(0, 1)
    ax.set_xlim(-lim, lim)
    ax.legend(prop={'size': 15})
    ax.set_xlabel('$y$, an annual maximum', fontsize=15)
    ax.set_ylabel('$f_{GEV}(y|1,1,\zeta)$, the probability density\n'
                  'function of the GEV distribution', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    # plt.show()
    filename = "{}/{}".format(VERSION_TIME, "gev plot big")
    StudyVisualizer.savefig_in_results(filename, transparent=True)
    plt.close()



def gev_plot_big_non_stationary_location():
    lim = 5
    x = np.linspace(-lim, lim, 100)
    scale, shape = 1, 0
    locs = [0.5, 1, 2]
    inverse_loc_with_scale = True
    colors = ['red','k', 'green']
    greek_leeter = ' $\{}_1'.format('mu' if not inverse_loc_with_scale else 'sigma')
    plt.title('Density for the distribution of Y(t) with different{}$'.format(greek_leeter))
    template = greek_leeter + '{} 0$'
    for loc, color in zip(locs, colors):
        if loc == locs[1]:
            sign_str = '='
        elif loc == locs[2]:
            sign_str = '>'
        else:
            sign_str = '<'
        label = template.format(sign_str)
        if inverse_loc_with_scale:
            loc, scale = scale, loc
        print(loc, scale, shape)
        y = r.dgev(x, loc, scale, shape)
        plt.plot(x, y, label=label, linewidth=5, color=color)
    plt.legend(prop={'size': 20})
    plt.xlabel('$y$', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.show()


def max_stable_plot():
    power_n_list = [1, 2, 3]
    fig, axes = plt.subplots(1, len(power_n_list), sharey='row')
    fig.subplots_adjust(hspace=0.4, wspace=0.4, )
    for j, power_n in enumerate(power_n_list):
        ax = axes[j]
        lim = 10 ** (power_n)
        x = np.linspace(0, lim, 100)
        loc, scale, shape = 1, 1, 1
        for n in [10 ** i for i in range(power_n)]:
            label = 'n={}'.format(n)
            y = np.array(r.pgev(x, loc, scale, shape))
            y **= n
            ax.plot(x, y, label=label)
        ax.legend(loc=4)
        ax.set_xlabel('$z$')
        if j == 0:
            ax.set_ylabel('$P(\\frac{ \max{(Z_1, ..., Z_n)} - b_n}{a_n} \leq z)$')
    plt.show()


def max_stable_plot_v2():
    power_n = 3
    fig, ax = plt.subplots(1, 1, sharey='row')
    loc, scale, shape = 1, 1, 1
    for n in [10 ** i for i in range(power_n)]:
        lim_left, lim_right = n, n * 10
        x = np.linspace(lim_left, lim_right, 100)
        label = 'n={}'.format(n)
        y = np.array(r.pgev(x, loc, scale, shape))
        y **= n
        ax.semilogx(x, y, label=label)
        ax.legend(loc=4)
        ax.set_xlabel('$z$')
        ax.set_ylabel('$P(\\frac{ \max{(Z_1, ..., Z_n)} - b_n}{a_n} \leq z)$')
    plt.show()


def quantile_function_plot():
    p = np.linspace(5e-1, 1e-5, 100)
    loc, scale = 2, 1
    shapes = [-1, 0, 1]
    for shape in shapes:
        label = '$\zeta= {} $'.format(shape)
        funct = [r.qgev, r.qgpd][0]
        y = funct(1 - p, loc, scale, shape)
        plt.loglog(1 / p, y, basex=10, label=label)
    plt.legend()
    plt.xlabel('$1/p$')
    plt.ylabel('$q(1- p|\mu, \sigma, \zeta)$')
    plt.show()


if __name__ == '__main__':
    # gev_plot()
    gev_plot_big()
    # gev_plot_big_non_stationary_location()
    # max_stable_plot()
    # quantile_function_plot()
    # max_stable_plot_v2()
