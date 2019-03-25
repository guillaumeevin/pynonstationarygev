


import numpy as np
import matplotlib.pyplot as plt

from extreme_estimator.extreme_models.utils import r, set_seed_r


def convergence_quantile_function(zoom=False):

    # Convergence of the quantile function
    eps = 1e-3
    left = 0.9 if zoom else eps
    p = np.linspace(left, 1-eps, 100)

    # n_list = range(1, 10)
    n_list = [1, 10, 100]
    for n in n_list:
        v = r.qexp(np.power(p, 1/n))
        plt.plot(p, v, label=n)
        quantile_perfect_gev = np.array(r.qgev(p, shape=0))
        quantile_perfect_gev += np.log(n)
        plt.plot(p, quantile_perfect_gev, label='gev' + str(n))
    plt.legend()
    plt.show()
    # remark: convergence is from above, the block maxima quantiles are above its correspond quantile gev distribtuion
    # this contradicts the hypothesis the issue I raised in "Simulation to understand quantile gap"


def convergence_repartition_function():
    # Convergence of the repartition function
    lim = 2
    x = np.linspace(-lim, lim, 100)
    plt.plot(x, r.pgev(x, shape=0), label='gev')
    for n in range(1, 20, 5):
        v = (np.power(r.pexp(x + np.log(n)), n))
        plt.plot(x, v, label=n)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    set_seed_r(seed=21)
    convergence_quantile_function(zoom=True)