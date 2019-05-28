from collections import OrderedDict

from cached_property import cached_property

from experiment.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall
from experiment.meteo_france_data.visualization.study_visualization.main_study_visualizer import \
    ALL_ALTITUDES, ALL_ALTITUDES_WITH_20_STATIONS_AT_LEAST
from extreme_estimator.estimator.full_estimator.abstract_full_estimator import \
    FullEstimatorInASingleStepWithSmoothMargin
from extreme_estimator.extreme_models.margin_model.linear_margin_model import LinearAllParametersAllDimsMarginModel, \
    LinearLocationAllDimsMarginModel, LinearShapeAllDimsMarginModel
from extreme_estimator.extreme_models.max_stable_model.abstract_max_stable_model import CovarianceFunction
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.uniform_normalization import \
    BetweenZeroAndOneNormalization
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations
from test.test_utils import load_test_max_stable_models
from utils import get_display_name_from_object_type

DATA_PATH = r'/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/Johan_data/PrecipitationsAvalanches_MaxPrecipit_ParPoste_ParHiver_traites.xls'

import pandas as pd


class ComparisonAnalysis(object):

    def __init__(self, altitude=900, nb_border_data_to_remove=0, normalize_observations=True, margin=150,
                 transformation_class=BetweenZeroAndOneNormalization, exclude_some_massifs_from_the_intersection=False):
        self.exclude_some_massifs_from_the_intersection = exclude_some_massifs_from_the_intersection
        self.normalize_observations = normalize_observations
        self.altitude = altitude
        self.margin = margin
        self.transformation_class = transformation_class
        self.nb_border_data_to_remove = nb_border_data_to_remove
        self.year_min = 1958 + nb_border_data_to_remove
        self.year_max = 2004 - nb_border_data_to_remove

    ##################### STATION ATTRIBUTES ############################

    def load_main_df_for_altitude(self):
        df = pd.read_excel(DATA_PATH, sheet_name='max alpes 2500m presentes')
        df = df.iloc[:78]
        ind_altitude = self.altitude - self.margin < df['ALTITUDE']
        ind_altitude &= df['ALTITUDE'] <= self.altitude + self.margin
        df = df.loc[ind_altitude]  # type: pd.DataFrame
        # Remove dulpicate for commune, Pellafol we should keep the first, i.e. 930 which has more data than the other
        df.drop_duplicates(subset='COMMUNE', inplace=True)
        df.set_index('COMMUNE', inplace=True)
        df = df.iloc[:, 3:]
        # Get values
        df_values = self.get_values(df)
        # Keep only stations who have not any Nan values
        ind = ~df_values.isna().any(axis=1)
        df = df.loc[ind]
        return df

    def load_main_df_for_altitude_and_good_massifs(self):
        df = self.load_main_df_for_altitude().copy()
        # Keep only the massif that also belong to the study (so that the normalization are roughly comparable)
        ind_massif = df['MASSIF_PRA'].isin(self.intersection_massif_names)
        df = df.loc[ind_massif]
        # Keep only one station per massif, to have the same number of points (the first by default)
        df = df.drop_duplicates(subset='MASSIF_PRA')
        return df

    @property
    def stations_coordinates(self):
        df = self.load_main_df_for_altitude_and_good_massifs()
        df = df.loc[:, ['LAMBERTX', 'LAMBERTY']]
        df.rename({'LAMBERTX': AbstractCoordinates.COORDINATE_X,
                   'LAMBERTY': AbstractCoordinates.COORDINATE_Y}, axis=1, inplace=True)
        coordinates = AbstractSpatialCoordinates.from_df(df, transformation_class=self.transformation_class)
        return coordinates

    @property
    def stations_observations(self):
        df = self.load_main_df_for_altitude_and_good_massifs()
        df = self.get_values(df)
        obs = AbstractSpatioTemporalObservations(df_maxima_gev=df)
        if self.normalize_observations:
            obs.normalize()
        return obs

    def get_values(self, df):
        df = df.iloc[:, 7:]
        df.columns = df.columns.astype(int)
        df = df.loc[:, self.year_min:self.year_max]
        return df

    @property
    def station_dataset(self):
        dataset = AbstractDataset(observations=self.stations_observations,
                                  coordinates=self.stations_coordinates)
        return dataset

    @property
    def massif_names(self):
        df = self.load_main_df_for_altitude()
        return list(set(df['MASSIF_PRA']))

    ##################### STUDY ATTRIBUTES ############################

    @cached_property
    def study(self):
        # Build the study for the same years
        return SafranSnowfall(altitude=self.altitude, nb_consecutive_days=1, year_min=self.year_min,
                              year_max=self.year_max + 1)

    @cached_property
    def intersection_massif_names(self):
        intersection_of_massif_names = list(set(self.massif_names).intersection(set(self.study.study_massif_names)))
        diff_due_to_wrong_names = set(self.massif_names) - set(self.study.study_massif_names)
        assert not diff_due_to_wrong_names, diff_due_to_wrong_names

        # remove on purpose some massifs (to understand if it the massifs that change the results or the year that were removed)
        # this created big differences in the results for altitude=900m margin=150m and nb=2
        # maybe this is due to a difference between the massif coordinate and the station (that belong to the massif) coordinate
        # or this might be due to a big difference between the observations
        if self.exclude_some_massifs_from_the_intersection:
            massifs_to_remove = ['Mercantour']
            intersection_of_massif_names = list(set(intersection_of_massif_names) - set(massifs_to_remove))

        return intersection_of_massif_names

    def study_coordinates(self, use_study_coordinate_with_latitude_and_longitude=True):
        # Build coordinate, from two possibles dataframes for the coordinates
        if use_study_coordinate_with_latitude_and_longitude:
            df = self.study.df_massifs_longitude_and_latitude
        else:
            df = self.study.load_df_centroid()
        coordinate = AbstractSpatialCoordinates.from_df(df=df.loc[self.intersection_massif_names],
                                                        transformation_class=self.transformation_class)
        return coordinate

    @property
    def study_observations(self):
        # Get the observations
        observations = self.study.observations_annual_maxima
        maxima_gev_of_interest = observations.df_maxima_gev.loc[self.intersection_massif_names]
        observations.df_maxima_gev = maxima_gev_of_interest
        if self.normalize_observations:
            observations.normalize()
        return observations

    @property
    def study_dataset_latitude_longitude(self):
        dataset = AbstractDataset(observations=self.study_observations,
                                  coordinates=self.study_coordinates(
                                      use_study_coordinate_with_latitude_and_longitude=True))
        return dataset

    @property
    def study_dataset_lambert(self):
        dataset = AbstractDataset(observations=self.study_observations,
                                  coordinates=self.study_coordinates(
                                      use_study_coordinate_with_latitude_and_longitude=False))
        return dataset

    # After a short analysis (run df_altitude to check) we decided on the altitude
    # 900 and 1200 seems to be the best altitudes

    def load_main_df(self):
        df = pd.read_excel(DATA_PATH, sheet_name='max alpes 2500m presentes')
        df = df.iloc[:78, 4:]
        return df

    def reduce_altitude(self, altitude=900) -> pd.Series:
        df = self.load_main_df()
        margin = 150
        ind_altitude = altitude - margin < df['ALTITUDE']
        ind_altitude &= df['ALTITUDE'] <= altitude + margin
        df = df.loc[ind_altitude]
        # Put all the result into an ordered dict
        d = OrderedDict()
        # Number of stations
        d['Nb stations'] = len(df)
        # Number of massifs
        d['Nb massifs'] = len(set(df['MASSIF_PRA']))

        df_values = df.iloc[:, 7:]
        df_values_from_1958 = df_values.iloc[:, 13:]
        # Mean number of non-Nan values
        d['% of Nan'] = df_values_from_1958.isna().mean().mean()
        # Number of lines with only Nan
        d['Lines w Nan'] = df_values_from_1958.isna().all(axis=1).sum()
        return pd.Series(d)

    def altitude_short_analysis(self):
        altitudes = ALL_ALTITUDES
        df = pd.concat([self.reduce_altitude(altitude) for altitude in altitudes], axis=1)
        df = df.transpose()
        df.index = altitudes
        # WIth the observation, the altitude 1200 seems the best
        # 1200           nb_stations:23          nb_massifs:15
        # I should try a fit

        # Finally I might prefer the altitude 900, which seems to have less missing values
        print(df)

    ##################### COMPARE THE TWO DATASETS BY FITTING THE SAME MODEL ############################

    def spatial_comparison(self, margin_model_class):
        max_stable_models = load_test_max_stable_models(default_covariance_function=CovarianceFunction.powexp)
        for max_stable_model in [max_stable_models[1], max_stable_models[-2]]:
            print('\n\n', get_display_name_from_object_type(type(max_stable_model)))
            for dataset in [self.station_dataset, self.study_dataset_lambert]:
                margin_model = margin_model_class(coordinates=dataset.coordinates)
                estimator = FullEstimatorInASingleStepWithSmoothMargin(dataset=dataset,
                                                                       margin_model=margin_model,
                                                                       max_stable_model=max_stable_model)
                estimator.fit()
                print(estimator.margin_function_fitted.coef_dict)
                # print(estimato)


def choice_of_altitude_and_nb_border_data_to_remove_to_get_data_without_nan():
    for margin in [50, 100, 150, 200, 250, 300][2:3]:
        for altitude in [900, 1200, 1800][-1:]:
            for nb in range(1, 15):
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
            print('nb:', nb, comparison.intersection_massif_names)
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

if __name__ == '__main__':
    # run_comparison_for_optimal_parameters_for_altitude_900()
    choice_of_altitude_and_nb_border_data_to_remove_to_get_data_without_nan()
