from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusDepth
import matplotlib.pyplot as plt

study = CrocusDepth(altitude=1500)
years = []
height = []
for year, days in study.year_to_days.items():
    i = days.index(str(year+1) + '-04-01')
    a = study.year_to_daily_time_serie_array[year]
    j = study.study_massif_names.index('Aravis')
    h = a[i, j]
    print(h)
    height.append(h)
    years.append(year)
plt.plot(years, height)
plt.show()