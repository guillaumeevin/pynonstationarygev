import numpy as np
import matplotlib.pyplot as plt

from extreme_fit.model.utils import r, set_seed_r


def snowfall_plot(flip=False):
    set_seed_r(seed=21)
    mean, sd = 200, 50
    lim = 200
    x = np.linspace(max(0, mean-lim), mean+lim, 1000)
    y = r.dnorm(x, mean=mean, sd=sd)

    if flip:
        plt.plot(y, x)
    else:
        plt.plot(x, y)
        plt.legend()
        plt.xlabel('snowfall S in mm')
        plt.ylabel('P(S)')

    level_to_color = {0.99: 'r',
                      0.5: 'g'}
    for level, color in level_to_color.items():
        quantile = r.qnorm(p=level, mean=mean, sd=sd)
        print(level, color, quantile)
        if flip:
            plt.plot(r.dnorm(quantile, mean=mean, sd=sd), quantile, color + 'o')
        else:
            plt.plot(quantile, r.dnorm(quantile, mean=mean, sd=sd), color + 'o')

    # Place the sample
    if not flip:
        n = 50
        plt.plot(r.rnorm(n=n, mean=mean, sd=sd), np.zeros(n), 'ko')

    plt.show()


if __name__ == '__main__':
    snowfall_plot(flip=True)
