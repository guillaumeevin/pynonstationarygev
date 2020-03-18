import matplotlib.pyplot as plt
from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal
from experiment.meteo_france_data.scm_models_data.crocus.crocus_variables import AbstractSnowLoadVariable

study = CrocusSnowLoadTotal(altitude=1800)
year = 1978
vercors_idx = study.study_massif_names.index('Vercors')
daily_time_series_vercors_fory_year_of_interest = study.year_to_daily_time_serie_array[year][:, vercors_idx]
days = [d[5:] for d in study.year_to_days[year]]
ax = plt.gca()
x = list(range(len(days)))
ax.plot(x, daily_time_series_vercors_fory_year_of_interest, linewidth='5')
ticks_date = ['{:02d}-01'.format(e) for e in [8, 9, 10, 11, 12] + list(range(1, 8))][::2]
ticks_index = [days.index(d) for d in ticks_date]
ticks = [x[t] for t in ticks_index]
labels = ['/'.join(t.split('-')[::-1]) + '/' + (str(year-1) if i < 3 else str(year)) for i, t in enumerate(ticks_date)]
ax.grid()
plt.xticks(ticks=ticks_index, labels=labels)
fontsize = 20
ax.set_xlabel("Date".format(year), fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_ylabel("GSL ({})".format(AbstractSnowLoadVariable.UNIT), fontsize=fontsize)
plt.show()
