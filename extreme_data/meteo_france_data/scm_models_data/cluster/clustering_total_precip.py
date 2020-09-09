from extreme_data.meteo_france_data.scm_models_data.cluster.abstract_cluster import AbstractCluster
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranPrecipitation1Day
import matplotlib.pyplot as plt

from root_utils import classproperty


class TotalPrecipCluster(AbstractCluster):




    @classproperty
    def massif_name_to_cluster_id(cls):
        return {
            'Chablais': 1,
            'Aravis': 1,
            'Mont-Blanc': 1,
            'Bauges': 1,
            'Beaufortain': 1,
            'Haute-Tarentaise': 2,
            'Chartreuse': 1,
            'Belledonne': 1,
            'Maurienne': 2,
            'Vanoise': 2,
            'Haute-Maurienne': 3,
            'Grandes-Rousses': 2,
            'Thabor': 3,
            'Vercors': 1,
            'Oisans': 2,
            'Pelvoux': 2,
            'Queyras': 3,
            'Devoluy': 2,
            'Champsaur': 2,
            'Parpaillon': 3,
            'Ubaye': 3,
            'Haut_Var-Haut_Verdon': 4,
            'Mercantour': 4
        }

    @classproperty
    def cluster_id_to_cluster_name(cls):
        return {
            1: 'North',
            2: 'Central',
            3: 'South',
            4: 'Extreme south'
        }


def massif_name_to_total_precipitation(altitude=2100):
    """
    We split the Alps in 4 regions based on mean total precipitation at 2100 m
    Extreme South: 2 massifs in the South
    South: 2 massifs with mean total precipitation < 1200mm
    Central: 12 massifs with 1200mm < mean < 1500mm
    North: 7 massifs with mean > 1500mm
    :param altitude:
    :return:
    """
    study = SafranPrecipitation1Day(altitude=altitude)
    total_precipitation = study.all_daily_series.sum(axis=0) / study.nb_years
    massif_name_to_total_precip = dict(zip(study.study_massif_names, total_precipitation))
    print('\nNorth ALps')
    for m, t in massif_name_to_total_precip.items():
        if t > 1500:
            print(m)
    print('\nSOuth ALps')
    for m, t in massif_name_to_total_precip.items():
        if t < 1200:
            print(m)
    print(massif_name_to_total_precip)
    vmin, vmax = min(total_precipitation), max(total_precipitation)
    study.visualize_study(massif_name_to_value=massif_name_to_total_precip, vmin=vmin, vmax=vmax,
                          add_colorbar=True, replace_blue_by_white=True)
    plt.show()
    return massif_name_to_total_precip


if __name__ == '__main__':
    d = massif_name_to_total_precipitation()

    print(d)
