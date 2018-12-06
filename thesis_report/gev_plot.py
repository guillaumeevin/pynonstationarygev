import numpy as np
import matplotlib.pyplot as plt

from extreme_estimator.extreme_models.utils import r


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
    plt.ylabel('$p(y|\mu, \sigma, \zeta)$')
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
        for n in [10**i for i in range(power_n)]:
            label = 'n={}'.format(n)
            y = np.array(r.pgev(x, loc, scale, shape))
            y **= n
            ax.plot(x, y, label=label)
        ax.legend(loc=4)
        ax.set_xlabel('$z$')
        if j == 0:
            ax.set_ylabel('$P(\\frac{ \max{(Z_1, ..., Z_n)} - b_n}{a_n} \leq z)$')
    plt.show()


if __name__ == '__main__':
#     gev_plot()
    max_stable_plot()
