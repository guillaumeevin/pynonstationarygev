from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies

if __name__ == '__main__':
    studies = AltitudesStudies(study_class=SafranSnowfall1Day,
                     altitudes=[600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600])
    for std in [True, False]:
        studies.plot_mean_maxima_against_altitude(std=std)