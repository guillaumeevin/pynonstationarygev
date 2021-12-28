import matplotlib.pyplot as plt
import numpy as np

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall
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

def histogram_for_gev_snowfall():
    import matplotlib.pyplot as plt
    ax = plt.gca()
    study_class = SafranSnowfall
    study = study_class(altitude=1800)
    s = study.observations_annual_maxima.df_maxima_gev.loc['Chartreuse']
    x_gev = s.values
    gev_params = fitted_stationary_gev(x_gev)
    print(gev_params.return_level(return_period=100))
    samples = gev_params.sample(10000)
    nb = 12
    epsilon = 0.0
    x, bins, p = ax.hist(samples, bins=[25 * i for i in range(10)],
                         color='white', edgecolor='grey', density=True, stacked=True,
                         linewidth=3, label='histogram')
    for item in p:
        item.set_height((item.get_height() / sum(x)))
    print(gev_params)
    x_density = np.linspace(0.0, 250, 1000)
    y_density = gev_params.density(x_density)
    factor = max([item.get_height() for item in p]) / max(y_density)
    print(max(y_density))
    y_density = [k * factor for k in y_density]
    print(max(y_density))
    ax.plot(x_density, y_density, linewidth=2, color='k', linestyle='--',
            label='probability density function')
    ax.set_xlabel('Annual maximum of daily snowfall (kg m$^{-2}$)', fontsize=15)
    ax.set_ylabel('Probability', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_yticks([0.1 * j for j in range(6)])
    ax.set_xticks([50 * j for j in range(6)])
    ax.set_xlim([0, 250])
    ax.set_ylim([0, 0.47])
    ax.legend(prop={'size': 12})


def annual_maxima_time_series_snowfall():
    import matplotlib.pyplot as plt
    ylim = 150
    ax = plt.gca()
    study_class = SafranSnowfall
    study = study_class(altitude=1800)
    massif_name = 'Chartreuse'
    y = study.massif_name_to_annual_maxima[massif_name]
    x = study.ordered_years
    ax.plot(x, y, color='green', marker='o')

    print(x[-1])

    ax.set_xlabel('Year', fontsize=15)
    ax.set_ylabel('Annual maximum of snowfall (kg m$^{-2}$)', fontsize=15)

    # plt.xticks(rotation=70)
    # ax.tick_params(axis='x', which='major', labelsize=6)
    # ax.set_yticks([0.1 * j for j in range(6)])
    ax.set_xticks(x[::10])
    ax.set_ylim([0, 220])
    ax.set_xlim([x[0], x[-1]])


def daily_time_series_snowfall():
    import matplotlib.pyplot as plt
    ylim = 150
    ax = plt.gca()
    study_class = SafranSnowfall
    study = study_class(altitude=1800, year_max=1962)
    massif_name = 'Chartreuse'
    y = study.massif_name_to_daily_time_series[massif_name]
    x = list(range(len(y)))
    xlabels = study.all_days
    ax.plot(x, y, label='daily values', color='grey')

    # Draw the limit between each years
    ymax = study.massif_name_to_annual_maxima[massif_name]
    xi_for_ymax = []
    count = 0
    for xi, xlabel, yi in zip(x, xlabels, y):
        if yi == ymax[count]:
            xi_for_ymax.append(xi)

        print(xlabel)
        if xlabel[-5:-3] == '08' and xlabel[-2:] == '01':
            if xi > 0:
                ax.vlines(xi, 0, ylim, linestyles='--', color='k', linewidth=2)

                count += 1
            print(len(xi_for_ymax), count)
            assert len(xi_for_ymax) == count

    # Plot a dot on each max
    ax.plot(xi_for_ymax, ymax, marker='o', linestyle=None, label='annual maxima',
            color='green')

    ax.set_xlabel('Date', fontsize=15)
    ax.set_ylabel('Daily value of snowfall (kg m$^{-2}$)', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)

    # Extract the xi that correspond to the first of a month

    xi_list = []
    xi_labels = []
    for xi, xlabel in zip(x, xlabels):
        if xlabel[-2:] == '01':
            xi_list.append(xi)
            xi_labels.append(xlabel)
    plt.xticks(rotation=70)
    ax.tick_params(axis='x', which='major', labelsize=6)
    ax.set_xticks(xi_list[::2])
    ax.set_xticklabels(xi_labels[::2])
    # ax.set_yticks([0.1 * j for j in range(6)])
    # ax.set_xticks([50 * j for j in range(6)])
    ax.set_xlim([0, 1500])
    ax.set_ylim([0, ylim])
    ax.legend(prop={'size': 10})


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
    # histogram_for_gev()
    # histogram_for_normal()

    # Snowfall for the thesis presentation
    # histogram_for_gev_snowfall()
    # daily_time_series_snowfall()
    annual_maxima_time_series_snowfall()
    plt.show()
