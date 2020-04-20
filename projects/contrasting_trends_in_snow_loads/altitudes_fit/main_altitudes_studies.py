from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from projects.contrasting_trends_in_snow_loads.altitudes_fit.altitudes_studies import AltitudesStudies


def main_plots():
    altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]
    study_class = SafranSnowfall1Day
    studies = AltitudesStudies(study_class, altitudes)
    # massifs_names = ['Vercors']
    # studies.plot_maxima_time_series(massif_names=massifs_names)
    studies.plot_maxima_time_series()


if __name__ == '__main__':
    main_plots()
