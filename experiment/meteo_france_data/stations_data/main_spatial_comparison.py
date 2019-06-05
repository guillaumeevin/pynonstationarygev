from experiment.meteo_france_data.stations_data.comparison_analysis import ComparisonAnalysis
from extreme_estimator.extreme_models.margin_model.linear_margin_model import LinearAllParametersAllDimsMarginModel
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.uniform_normalization import \
    BetweenZeroAndOneNormalization
from utils import get_display_name_from_object_type


def choice_of_altitude_and_nb_border_data_to_remove_to_get_data_without_nan():
    for margin in [50, 100, 150, 200, 250, 300][2:3]:
        for altitude in [900, 1200, 1800][:1]:
            for nb in range(1, 4):
                s = ComparisonAnalysis(altitude=altitude, nb_border_data_to_remove=nb, margin=margin)
                print(margin, altitude, nb, 'nb massifs', len(s.intersection_massif_names), 'nb stations',
                      len(s.stations_observations), 'nb observations', s.stations_observations.nb_obs,
                      s.study_observations.nb_obs,
                      s.stations_coordinates.index)


def run_comparison_for_optimal_parameters_for_altitude_900():
    for nb in [0, 1, 2][:]:
        for transformation_class in [None, BetweenZeroAndOneNormalization][1:]:
            comparison = ComparisonAnalysis(altitude=900, nb_border_data_to_remove=nb, margin=150,
                                            exclude_some_massifs_from_the_intersection=nb == 2,
                                            transformation_class=transformation_class,
                                            normalize_observations=True)
            print('\n-----------\nnb:', nb, comparison.intersection_massif_names)
            # margin_model_classes = [LinearShapeAllDimsMarginModel, LinearLocationAllDimsMarginModel,
            #           LinearAllParametersAllDimsMarginModel]
            for margin_model_class in [LinearAllParametersAllDimsMarginModel]:
                print(get_display_name_from_object_type(margin_model_class))
                comparison.spatial_comparison(margin_model_class)


"""
Comparaison données de re-analysis et données de stations

J'ai utilisé le fichier "PrecipitationsAvalanches_MaxPrecipit_ParPoste_ParHiver_traites.xls"

Après des analyses avec la fonction 'choice_of_altitude_and_nb_border_data_to_remove_to_get_data_without_nan'
j'ai choisis de lancer mes analyses avec:
    -une altitude de 900m 
    -une margin de 150m (donc je selectionne toutes les stations entre 750m et 1050m). 
Je ne choisis que des stations qui ont des observations complètes sur toute la periode d'observation. 
et je m'asssure de n'avoir une seule station par massif (qui appartient à l intersection des massifs entre les study et les stations)

Souvent les observations manquantes se situaient dans les premières ou dans les dernières années
j'ai donc ajouté un parametre nb_to_remove_border qui enlever ces observations (à la fois pour les study et les stations).
Ce parametre entrainent donc des datasets avec moins d observations, mais avec plus de masssifs/stations

Par contre, dans le cas nb_to_remove=2, il y avait de grosses différences si j'incluais ou non le massif Mercantour
donc en tout attendant de mieux comprendre, j'ai prefere exclure ce massif dans ce cas

Dans tous les cas, nb_to_remove de 0 à 2
pour n'importe quel modele de marges
et pour un max stable BrownResnick ou ExtremalT
alors le signe des coefficient de marges selon les coordonées Lambert sont toujours les mêmes que l'on utilise les données 
de reanalysis ou les données de stations
"""


"""
A way to improve the analysis would be to have another altitude of reference with a lot of data
But for the other altitude, we have data issues because there is a Nan in the middle of the data
Instead of removing on the side, I should remove the years that concerns as much station from the same altitude level
I should find the "optimal" years to remove
Then I should find a way to remove the same years in the study
"""

def test_data():
    s = ComparisonAnalysis(altitude=900)
    df = s.load_main_df()
    print(df)
    print(df.columns)
    print(len(df))

if __name__ == '__main__':
    test_data()
    # run_comparison_for_optimal_parameters_for_altitude_900()
    # choice_of_altitude_and_nb_border_data_to_remove_to_get_data_without_nan()
