from extreme_data.meteo_france_data.adamont_data.adamont.adamont_crocus import AdamontSnowLoad
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_max_swe import CrocusSnowLoad2019


def simulation_parameters_from_study():
    """Fit a non-stationary GEV with a linear non-stationarity on a all the parameters"""
    study = CrocusSnowLoad2019(altitude=1500)
    gev_params = study.massif_name_to_stationary_gev_params['Vanoise']
    # {'loc': 2.525630227017153, 'scale': 1.1137803762152, 'shape': -0.10546443622078182}
    print(gev_params)


if __name__ == '__main__':
    simulation_parameters_from_study()