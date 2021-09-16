from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal

massif_names = ['Grandes-Rousses', 'Pelvoux']
altitudes = [2100, 2400]
study_class = CrocusSnowLoadTotal
for altitude in altitudes:
    study = study_class(altitude=altitude)
    for massif_name in massif_names:
        gev_params = study.massif_name_to_stationary_gev_params[massif_name]
        print('At ', altitude, ' m, for the massif', massif_name, 'the 50-year return level is:', round(gev_params.return_level(return_period=50), 2), 'kN m-2')