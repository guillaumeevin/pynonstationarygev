from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranSnowfall3Days, \
    SafranSnowfall5Days, SafranSnowfall7Days, SafranPrecipitation1Day, SafranPrecipitation3Days, \
    SafranPrecipitation5Days, SafranPrecipitation7Days
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies


def main_plots_moments():
    altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]
    # altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900]
    study_classes = [SafranSnowfall1Day, SafranSnowfall3Days, SafranSnowfall5Days, SafranSnowfall7Days][:2]
    study_classes = [SafranPrecipitation1Day, SafranPrecipitation3Days, SafranPrecipitation5Days,
                     SafranPrecipitation7Days][:]
    study_classes = [SafranPrecipitation1Day, SafranSnowfall1Day, SafranSnowfall3Days, SafranPrecipitation3Days][2:]

    for study_class in study_classes:

        studies = AltitudesStudies(study_class, altitudes, season=Season.winter_extended)
        # massifs_names = ['Vercors', 'Chartreuse', 'Belledonne']
        # studies.plot_mean_maxima_against_altitude(massif_names=massifs_names, show=True)
        # studies.plot_maxima_time_series()
        for std in [True, False][1:]:
            for change in [True, False, None]:
                studies.plot_mean_maxima_against_altitude(std=std, change=change)


if __name__ == '__main__':
    main_plots_moments()
