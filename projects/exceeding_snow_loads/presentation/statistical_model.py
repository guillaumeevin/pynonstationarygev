import matplotlib.pyplot as plt
import numpy as np

from extreme_fit.estimator.margin_estimator.utils import fitted_stationary_gev


def binomial_observation():
    marker_altitude_massif_name_for_paper1 = [
        ('magenta', 'stationary', [0.5, 0.5]),
        # ('darkmagenta', 'increase', [0.5, 0.8]),
        # ('mediumpurple', 'decrease', [0.5, 0.3]),
    ]
    ax = plt.gca()
    for color, label, l in marker_altitude_massif_name_for_paper1:
        before, after = l
        total_time = 60
        data_before = np.random.binomial(1, before, int(total_time / 3))
        data_after = np.random.binomial(1, after, int(2 * total_time / 3))
        data = np.concatenate([data_before, data_after], axis=0)
        time = list(range(total_time))
        ax.tick_params(axis='both', which='major', labelsize=15)
        fontsize = 20
        ax.set_xlabel('Time', fontsize=fontsize)
        ax.set_ylabel('Coin value', fontsize=fontsize)
        plt.yticks([0.0, 1.0], ['Heads', 'Tails'])
        ax.plot(time, data, color=color, label=label, linewidth=4)


def histogram_for_gev():
    import matplotlib.pyplot as plt
    from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal
    from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_variables import AbstractSnowLoadVariable
    from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusDepth
    ax = plt.gca()
    study_class = CrocusDepth
    study = study_class(altitude=900)
    s = study.observations_annual_maxima.df_maxima_gev.loc['Chartreuse']
    x_gev = s.values
    gev_params = fitted_stationary_gev(x_gev)
    print(gev_params.return_level(return_period=50))
    samples = gev_params.sample(10000)
    nb = 12
    epsilon = 0.0
    x, bins, p = ax.hist(samples, bins=[0.25 * i for i in range(10)],
                         color='white', edgecolor='grey', density=True, stacked=True,
                         linewidth=3)
    for item in p:
        item.set_height((item.get_height() / sum(x)))
    print(gev_params)
    # x = np.linspace(0.0, 10, 1000)
    # y = gev_params.density(x)
    # ax.plot(x, y, linewidth=5)
    ax.set_xlabel('Annual maximum of snow depth (m)', fontsize=15)
    ax.set_ylabel('Probability', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_yticks([0.1 * j for j in range(4)])
    ax.set_xticks([0.5 * j for j in range(5)])
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 0.36])


def histogram_for_normal():
    ax = plt.gca()
    linewidth = 5
    absisse = [0.25, 0.75]
    ax.bar(absisse, [0.6, 0.6], width=0.2, color='white', edgecolor='grey', linewidth=linewidth, bottom=-0.1)
    plt.xticks(absisse, ['Heads', 'Tail'])
    ax.set_yticks([0, 0.5])
    ax.set_xlabel('Coin value', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)

    ax.set_ylabel('Probability', fontsize=15)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 0.6])


if __name__ == '__main__':
    # binomial_observation()
    histogram_for_gev()
    # histogram_for_normal()
    plt.show()
