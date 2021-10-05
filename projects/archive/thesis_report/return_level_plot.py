import numpy as np
import matplotlib.pyplot as plt

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.utils import r



def return_level_plot_big():
    lim = 100
    ax = plt.gca()
    r_list = list(range(2, 11))
    # log_y_list = np.linspace(-1, 2, 100)
    # y_list = [np.exp(l) for l in log_y_list]
    # p = [np.exp(-y) for y in y_list]
    # return_period = [1 - 1/m for m in p]
    # print(p[0], p[-1])
    # q =
    # r_list = np.linspace(2, lim, 100)
    # p_list = [1 - 1/r for r in r_list]
    # xp = [np.log(-np.log(1 - p)) for p in p_list]
    loc, scale = 1, 1
    shapes = [-1, 0, 1]
    colors = ['tab:blue', 'tab:red', 'tab:green']
    for shape, color in zip(shapes, colors):
        label = '$\zeta= {} $'.format(shape)
        gev_params = GevParams(loc, scale, shape)
        # y = [-gev_params.quantile(g) for g in p]
        y = [gev_params.return_level(r) for r in r_list]
        print('\nshape:',shape)
        print(r_list)
        print(y)
        ax.plot(r_list, y, label=label, linewidth=5, color=color)
    ax.legend(prop={'size': 15})
    ax.set_xlabel('p the probability to exceed the return level', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_ylabel('$y_p$ the return level', fontsize=15)

    # ax2 = ax.twiny()
    # ax2.tick_params(labelsize=15)
    # ax2.set_xlim(ax.get_xlim())
    # ax2.set_xlabel('T the return period (year)', fontsize=15)

    labels = [str(int(e)) for e in ax.get_xticks()]
    ax.set_xticklabels(['1/{}'.format(l) for l in labels])
    plt.show()

if __name__ == '__main__':
    return_level_plot_big()
