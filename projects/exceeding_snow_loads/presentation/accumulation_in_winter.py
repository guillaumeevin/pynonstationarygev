import matplotlib.pyplot as plt
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal, CrocusDepth
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_variables import AbstractSnowLoadVariable
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day

ax = plt.gca()
fontsize = 20
altitude = 1800
# studies = [CrocusSnowLoadTotal(altitude=altitude), CrocusDepth(altitude=altitude)]
studies = [SafranSnowfall1Day(altitude=altitude), CrocusDepth(altitude=altitude)]
colors = ['black', 'grey']
for i, study in enumerate(studies):
    color = colors[i]
    if i == 1:
        ax = ax.twinx()

    year = 1978
    vercors_idx = study.study_massif_names.index('Vercors')
    daily_time_series_vercors_fory_year_of_interest = study.year_to_daily_time_serie_array[year][:, vercors_idx]
    days = [d[5:] for d in study.year_to_days[year]]
    x = list(range(len(days)))
    if i == 0:
        if isinstance(study, SafranSnowfall1Day):
            ylabel = 'snowfall (mm)'
        else:
            ylabel = 'ground snow load ({})'.format(AbstractSnowLoadVariable.UNIT)
    else:
        ylabel = 'snow depth (m)'
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.plot(x, daily_time_series_vercors_fory_year_of_interest, linewidth='5', color=color, label=ylabel)
    ticks_date = ['{:02d}-01'.format(e) for e in [8, 9, 10, 11, 12] + list(range(1, 8))][::2]
    ticks_index = [days.index(d) for d in ticks_date]
    ticks = [x[t] for t in ticks_index]
    labels = ['/'.join(t.split('-')[::-1]) + '/' + (str(year-1) if i < 3 else str(year)) for i, t in enumerate(ticks_date)]
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_ylim(bottom=0)
    if i == 0:
        ax.grid()
        plt.xticks(ticks=ticks_index, labels=labels)
        ax.set_xlabel("Date".format(year), fontsize=fontsize)
    ax.legend(prop={'size': 14})
plt.show()
