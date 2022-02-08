import matplotlib.pyplot as plt
import numpy as np

from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import Crocus, CrocusSnowLoadTotal
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_fit.estimator.margin_estimator.utils import fitted_stationary_gev
from root_utils import VERSION_TIME


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

def histogram_for_adjustment_coefficients(past=True):
    add_location, add_scale = 0, 0
    import matplotlib.pyplot as plt
    if not past:
        add_location -= 2
    ax = plt.gca()
    study_class = CrocusSnowLoadTotal
    study = study_class(altitude=1500)
    s = study.observations_annual_maxima.df_maxima_gev.loc['Vercors']
    x_gev = s.values
    for color, add_loc, add_sca in zip(["green", 'k'], [2, 0], [1, 0]):
        plot_histo(add_location + add_loc, add_scale + add_sca, ax, x_gev, color, linewidth=5)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel('Annual maximum of snow load (kN m$^{-2}$)', fontsize=15)
    # ax.set_yticks([0.1 * j for j in range(6)])
    # ax.set_xticks([50 * j for j in range(6)])
    # ax.set_xlim([0, 250])
    # ax.set_ylim([0, 0.5])
    # ax.legend(prop={'size': 12})

    filename = "{}/{}".format(VERSION_TIME, "gev histo {}".format(past))
    StudyVisualizer.savefig_in_results(filename, transparent=False, tight_pad={'h_pad': 0.1})
    plt.close()


def histogram_for_gev_snowfall(add_location=0, add_scale=0):
    import matplotlib.pyplot as plt
    ax = plt.gca()
    study_class = SafranSnowfall
    study = study_class(altitude=1800)
    s = study.observations_annual_maxima.df_maxima_gev.loc['Chartreuse']
    x_gev = s.values
    plot_histo(add_location, add_scale, ax, x_gev)
    ax.set_xlabel('Annual maximum of daily snowfall (kg m$^{-2}$)', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_yticks([0.1 * j for j in range(6)])
    ax.set_xticks([50 * j for j in range(6)])
    ax.set_xlim([0, 250])
    ax.set_ylim([0, 0.5])
    ax.legend(prop={'size': 12})

    filename = "{}/{}".format(VERSION_TIME, "gev histo {} {}".format(add_location, add_scale))
    StudyVisualizer.savefig_in_results(filename, transparent=False, tight_pad={'h_pad': 0.1})
    plt.close()


def histogram_for_gev_snow_load(add_location=0.0, add_scale=0.0):
    import matplotlib.pyplot as plt
    ax = plt.gca()
    study_class = CrocusSnowLoadTotal
    study = study_class(altitude=1500)
    s = study.observations_annual_maxima.df_maxima_gev.loc['Vanoise']
    x_gev = s.values
    plot_histo_bis(add_location, add_scale, ax, x_gev)
    ax.set_xlabel('Annual maximum of snow load (kN m$^{-2}$)', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_yticks([0.1 * j for j in range(5)])
    ax.set_xticks([2 * j for j in range(6)])
    ax.set_xlim([0, 11])
    ax.set_ylim([0, 0.4])

    ax.legend(prop={'size': 12})

    filename = "{}/{}".format(VERSION_TIME, "gev histo {} {}".format(add_location, add_scale))
    StudyVisualizer.savefig_in_results(filename, transparent=False, tight_pad={'h_pad': 0.1})
    plt.close()


def return_level_for_gev_snow_load():
    import matplotlib.pyplot as plt
    first = 6.084228523010236
    last = 5.411714800614831
    ax = plt.gca()

    x = [1959 + i for i in range(62)]
    x_scaled = [(e - 1959) / 61 for e in x]
    assert x_scaled[0] == 0
    assert x_scaled[-1] == 1
    y = [first + (last - first) * e for e in x_scaled]
    print(x)
    print(y)
    ax.plot(x, y, color='orange', linewidth=4)
    ax.set_ylabel('50-year return level of snow load (kN m$^{-2}$)', fontsize=15)
    ax.set_xlabel('Years', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_yticks([5 + j for j in range(3)])
    ax.set_xticks([1959 + 20 * j for j in range(4)])
    ax.set_xlim([1959, 2019])
    ax.set_ylim([4.5, 7])

    filename = "{}/{}".format(VERSION_TIME, "return level plot")
    StudyVisualizer.savefig_in_results(filename, transparent=False, tight_pad={'h_pad': 0.1})
    plt.close()



def plot_histo_bis(add_location, add_scale, ax, x_gev, edgecolor='grey', linewidth=3):
    gev_params = fitted_stationary_gev(x_gev)
    print('\nAdd some parameters: loc {} sclae {}'.format(add_location, add_scale))
    gev_params.location += add_location
    gev_params.scale += add_scale
    print('50 year return level', gev_params.return_level(return_period=50))
    samples = gev_params.sample(10000)
    if edgecolor == 'grey':
        maxi = 10
    else:
        maxi = 10


    linestyle = 'solid'
    if (edgecolor == 'k') and (add_location != 0):
        linestyle = 'dashed'

    nb = 12
    epsilon = 0.0
    bins = [maxi / 10 * i for i in range(10)]
    x, bins, p = ax.hist(samples, bins,
                         color='white', edgecolor=edgecolor, density=True, stacked=True,
                         linewidth=linewidth, label='histogram', linestyle=(linestyle))
    for item in p:
        item.set_height((item.get_height() / sum(x)))
    # print(gev_params)
    if edgecolor == 'grey':
        x_density = np.linspace(0.0, maxi, 1000)
        y_density = gev_params.density(x_density)
        factor = max([item.get_height() for item in p]) / max(y_density)
        # print(max(y_density))
        y_density = [k * factor for k in y_density]
        # print(max(y_density))
        ax.plot(x_density, y_density, linewidth=2, color='k', linestyle='--',
                label='probability density function')
    ax.set_ylabel('Probability', fontsize=15)


def plot_histo(add_location, add_scale, ax, x_gev, edgecolor='grey', linewidth=3):
    gev_params = fitted_stationary_gev(x_gev)
    print('\nAdd some parameters: loc {} sclae {}'.format(add_location, add_scale))
    gev_params.location += add_location
    gev_params.scale += add_scale
    print(gev_params.return_level(return_period=100))
    samples = gev_params.sample(10000)
    if edgecolor == 'grey':
        maxi = 250
    else:
        maxi = 10


    linestyle = 'solid'
    if (edgecolor == 'k') and (add_location != 0):
        linestyle = 'dashed'

    nb = 12
    epsilon = 0.0
    bins = [maxi / 10 * i for i in range(10)]
    x, bins, p = ax.hist(samples, bins,
                         color='white', edgecolor=edgecolor, density=True, stacked=True,
                         linewidth=linewidth, label='histogram', linestyle=(linestyle))
    for item in p:
        item.set_height((item.get_height() / sum(x)))
    print(gev_params)
    if edgecolor == 'grey':
        x_density = np.linspace(0.0, maxi, 1000)
        y_density = gev_params.density(x_density)
        factor = max([item.get_height() for item in p]) / max(y_density)
        print(max(y_density))
        y_density = [k * factor for k in y_density]
        print(max(y_density))
        ax.plot(x_density, y_density, linewidth=2, color='k', linestyle='--',
                label='probability density function')
    ax.set_ylabel('Probability', fontsize=15)


def daily_time_series_snowfall():
    import matplotlib.pyplot as plt
    ylim = 150
    ax = plt.gca()
    study_class = SafranSnowfall
    study = study_class(altitude=1800)
    massif_name = 'Chartreuse'
    y = study.massif_name_to_daily_time_series[massif_name]
    x = range(len(y))
    ax.plot(x, y, color='grey')

    xlabels = study.all_days
    xticks = []
    xtickslabels = []
    for xi, xlabel in zip(x, xlabels):
        print(xlabel)
        if xlabel[-7:] == '9-01-01':
            print(xlabel)
            xticks.append(xi)
            xtickslabels.append(xlabel)
    print(xticks)
    print(xtickslabels)



    plt.xticks(rotation=45)
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtickslabels)
    # ax.set_yticks([0.1 * j for j in range(6)])
    ax.set_ylim([0, 220])
    ax.set_xlim([x[0], x[-1]])

    ax.set_xlabel('Date', fontsize=15)
    ax.set_ylabel('Daily value of snowfall (kg m$^{-2}$)', fontsize=15)

    filename = '{}/daily time series'.format(VERSION_TIME)
    StudyVisualizer.savefig_in_results(filename, transparent=False, tight_pad={'h_pad': 0.1})
    plt.close()


def annual_maxima_time_series_snowfall():
    import matplotlib.pyplot as plt
    ylim = 150
    ax = plt.gca()
    study_class = SafranSnowfall
    study = study_class(altitude=1800)
    massif_name = 'Chartreuse'
    y = study.massif_name_to_annual_maxima[massif_name]
    x = study.ordered_years
    ax.plot(x, y, color='k', marker='o', linewidth=0)

    print(x[-1])

    ax.set_xlabel('Year', fontsize=15)
    ax.set_ylabel('Annual maxima of snowfall in the\nChartreuse massif at 1500 m (kg m$^{-2}$)', fontsize=15)

    # plt.xticks(rotation=70)
    # ax.tick_params(axis='x', which='major', labelsize=6)
    # ax.set_yticks([0.1 * j for j in range(6)])
    ax.set_xticks(x[::10])
    ax.set_ylim([0, 220])
    ax.set_xlim([x[0], x[-1]])
    filename = '{}/annual maxima'.format(VERSION_TIME)
    StudyVisualizer.savefig_in_results(filename, transparent=True, tight_pad={'h_pad': 0.1})
    plt.close()


def two_annual_maxima_time_series_snowfall():
    import matplotlib.pyplot as plt
    ylim = 150
    ax = plt.gca()
    study_class = SafranSnowfall
    massif_name = 'Chartreuse'
    for altitude, color in zip([900, 600], ['k', 'grey']):
        study = study_class(altitude=altitude)
        y = study.massif_name_to_annual_maxima[massif_name]
        x = study.ordered_years
        ax.plot(x, y, color=color, marker='o', linewidth=0, label='at {} m'.format(altitude))

    print(x[-1])

    ax.set_xlabel('Year', fontsize=15)
    ax.set_ylabel('Annual maxima of snowfall in the\nChartreuse massif (kg m$^{-2}$)', fontsize=15)

    # plt.xticks(rotation=70)
    # ax.tick_params(axis='x', which='major', labelsize=6)
    # ax.set_yticks([0.1 * j for j in range(6)])
    ax.set_xticks(x[::10])
    ax.legend()
    # ax.set_ylim([0, 220])
    ax.set_xlim([x[0], x[-1]])
    filename = '{}/annual maxima'.format(VERSION_TIME)
    StudyVisualizer.savefig_in_results(filename, transparent=True, tight_pad={'h_pad': 0.1})
    plt.close()

def daily_time_series_snowload_for_maxima_extraction():
    import matplotlib.pyplot as plt
    for year in [1959, 2009]:
        ax = plt.gca()
        study_class = CrocusSnowLoadTotal
        study = study_class(altitude=1500, year_min=year, year_max=year+10)
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
                    ax.vlines(xi, 0, 10, linestyles='--', color='k', linewidth=2)

                    count += 1
                print(len(xi_for_ymax), count)
                assert len(xi_for_ymax) == count

        # Plot a dot on each max
        ax.plot(xi_for_ymax, ymax, marker='o', linestyle=None, label='annual maxima',
                color='green')

        ax.set_xlabel('Date', fontsize=15)
        ax.set_ylabel('Daily value of accumulated snow load (kN m$^{-2}$)', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)

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
        # ax.set_xlim([0, 1500])
        # ax.set_ylim([0, ylim])
        ax.legend(prop={'size': 10})

        # Plot the years on top
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_yticks([])
        ax2.set_xticks(ax.get_xticks()[3::6])
        ax2.set_xticklabels(study.ordered_years)
        plt.show()


def daily_time_series_snowfall_for_maxima_extraction():
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
    ax.set_ylabel('Daily value of snowfall (kg m$^{-2}$)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)

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

    # Plot the years on top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_yticks([])
    ax2.set_xticks(ax.get_xticks()[3::6])
    ax2.set_xticklabels(study.ordered_years)

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

def histogram_for_selected_model_snow_load():
    x = [25, 60, 95, 130]
    width = 20
    percentages = [55, 22.5, 12.5, 10]
    labels = ['stationary', 'Linearity in $\mu$ and $\sigma$', 'Linearity in $\mu$', 'Linearity in $\sigma$']
    significance_percentage = [0, 16, 7.5, 4]

    ax = plt.gca()
    ax.bar(x, percentages, width, 0, color='grey')
    ax.bar(x, significance_percentage, width, 0, color='k')
    ax.set_xticks(x)
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('Selected model with the AIC')
    ax.set_xticklabels(labels)
    plt.show()

if __name__ == '__main__':
    # histogram_for_selected_model_snow_load()
    return_level_for_gev_snow_load()
    # binomial_observation()
    # histogram_for_gev()
    # histogram_for_normal()

    # daily_time_series_snowfall()

    # Snowfall for the thesis presentation
    # histogram_for_gev_snow_load()
    # for add_location in [-1, 1]:
    #     histogram_for_gev_snow_load(add_location=add_location)
    # f = 0.21
    # for add_scale in [-f, f]:
    #     histogram_for_gev_snow_load(add_scale=add_scale)

    # for past in [True, False]:
    #     histogram_for_adjustment_coefficients(past)

    # two_annual_maxima_time_series_snowfall()

    # daily_time_series_snowfall()
    # annual_maxima_time_series_snowfall()
    # daily_time_series_snowload_for_maxima_extraction()
    # plt.show()
