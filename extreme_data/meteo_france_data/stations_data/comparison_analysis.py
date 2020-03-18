from collections import OrderedDict
import numpy as np
from typing import List

from cached_property import cached_property

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    ALL_ALTITUDES
from extreme_fit.estimator.full_estimator.abstract_full_estimator import \
    FullEstimatorInASingleStepWithSmoothMargin
from extreme_fit.model.max_stable_model.abstract_max_stable_model import CovarianceFunction
from extreme_fit.model.max_stable_model.max_stable_models import ExtremalT, BrownResnick
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.uniform_normalization import \
    BetweenZeroAndOneNormalization
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations
from test.test_utils import load_test_max_stable_models
from root_utils import get_display_name_from_object_type

REANALYSE_STR = 'reanalyse'
ALTITUDE_COLUMN_NAME = 'ALTITUDE'
MASSIF_COLUMN_NAME = 'MASSIF_PRA'
STATION_COLUMN_NAME = 'STATION'

DATA_PATH = r'/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/Johan_data/PrecipitationsAvalanches_MaxPrecipit_ParPoste_ParHiver_traites.xls'

import pandas as pd


class ComparisonAnalysis(object):

    def __init__(self, altitude=900, nb_border_data_to_remove=0, normalize_observations=True, margin=150,
                 transformation_class=BetweenZeroAndOneNormalization, exclude_some_massifs_from_the_intersection=False,
                 keep_only_station_without_nan_values=True,
                 one_station_per_massif=True):
        self.keep_only_station_without_nan_values = keep_only_station_without_nan_values
        self.one_station_per_massif = one_station_per_massif
        self.exclude_some_massifs_from_the_intersection = exclude_some_massifs_from_the_intersection
        self.normalize_observations = normalize_observations
        self.altitude = altitude
        self.margin = margin
        self.transformation_class = transformation_class
        self.nb_border_data_to_remove = nb_border_data_to_remove
        self.year_min = 1958 + nb_border_data_to_remove
        self.year_max = 2004 - nb_border_data_to_remove

    ##################### STATION ATTRIBUTES ############################

    def load_main_df(self):
        # this sheet name: Mean metrics 24.666666666666668 for the sheet name it was worse Mean metrics 36.022222222222226
        df = pd.read_excel(DATA_PATH, sheet_name='max alpes')
        df = df.iloc[:78]

        ind_altitude = self.altitude - self.margin < df[ALTITUDE_COLUMN_NAME]
        ind_altitude &= df[ALTITUDE_COLUMN_NAME] <= self.altitude + self.margin
        df = df.loc[ind_altitude]  # type: pd.DataFrame
        # Remove dupicate for commune, Pellafol we should keep the first, i.e. 930 which has more data than the other
        df.drop_duplicates(subset='COMMUNE', inplace=True)
        df.set_index('COMMUNE', inplace=True)
        df = df.iloc[:, 3:]
        # Get values
        df_values = self.get_values(df)
        if self.keep_only_station_without_nan_values:
            # Keep only stations who have not any Nan values
            ind = ~df_values.isna().any(axis=1)
            df = df.loc[ind]
        return df

    def load_main_df_stations_that_belong_to_intersection_massifs(self):
        df = self.load_main_df().copy()
        # Keep only the massif that also belong to the study (so that the normalization are roughly comparable)
        ind_massif = df['MASSIF_PRA'].isin(self.intersection_massif_names)
        return df.loc[ind_massif]

    def load_main_df_one_station_per_massif(self):
        df = self.load_main_df_stations_that_belong_to_intersection_massifs()
        # Keep only one station per massif, to have the same number of points (the first by default)
        df = df.drop_duplicates(subset='MASSIF_PRA')
        # Sort all the DataFrame so that the massif order correspond
        df['MASSIF_IDX'] = [self.intersection_massif_names.index(m) for m in df['MASSIF_PRA']]
        df = df.sort_values(['MASSIF_IDX'])
        df.drop(labels='MASSIF_IDX', axis=1, inplace=True)
        return df

    @property
    def load_df_stations(self):
        if self.one_station_per_massif:
            return self.load_main_df_one_station_per_massif()
        else:
            return self.load_main_df_station_intersection_clean()

    @property
    def stations_coordinates(self):
        df = self.load_main_df()
        df = df.loc[:, ['LAMBERTX', 'LAMBERTY']]
        df.rename({'LAMBERTX': AbstractCoordinates.COORDINATE_X,
                   'LAMBERTY': AbstractCoordinates.COORDINATE_Y}, axis=1, inplace=True)
        coordinates = AbstractSpatialCoordinates.from_df(df, transformation_class=self.transformation_class)
        return coordinates

    @property
    def stations_observations(self):
        df = self.load_main_df()
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
        df = self.load_main_df()
        return list(set(df['MASSIF_PRA']))

    ##################### STUDY ATTRIBUTES ############################

    @cached_property
    def study(self):
        # Build the study for the same years
        return SafranSnowfall(altitude=self.altitude, nb_consecutive_days=3, year_min=self.year_min,
                              year_max=self.year_max + 1)

    @property
    def nb_massifs(self):
        return len(self.intersection_massif_names)

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
    def study_dataset_latitude_longitude(self) -> AbstractDataset:
        dataset = AbstractDataset(observations=self.study_observations,
                                  coordinates=self.study_coordinates(
                                      use_study_coordinate_with_latitude_and_longitude=True))
        return dataset

    @property
    def study_dataset_lambert(self) -> AbstractDataset:
        dataset = AbstractDataset(observations=self.study_observations,
                                  coordinates=self.study_coordinates(
                                      use_study_coordinate_with_latitude_and_longitude=False))
        return dataset

    # After a short analysis (run df_altitude to check) we decided on the altitude
    # 900 and 1200 seems to be the best altitudes

    def reduce_altitude(self, altitude=900) -> pd.Series:
        df = pd.read_excel(DATA_PATH, sheet_name='max alpes 2500m presentes')
        df = df.iloc[:78, 4:]
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

    ################## MERGING SOME ATTRIBUTES ###################
    def load_main_df_station_intersection_clean(self):
        assert not self.one_station_per_massif
        df = self.load_main_df_stations_that_belong_to_intersection_massifs().loc[:, ['MASSIF_PRA', 'ALTITUDE']]
        df_coord = self.stations_coordinates.df_all_coordinates
        df_obs = self.stations_observations.df_maxima_gev
        return pd.concat([df, df_coord, df_obs], axis=1)

    def load_main_df_study_intersection_clean(self):
        assert not self.one_station_per_massif
        df_coord = self.study_dataset_lambert.coordinates.df_all_coordinates
        df_obs = self.study_observations.df_maxima_gev
        study_index = pd.Series(df_coord.index) + ' ' + REANALYSE_STR
        df = pd.DataFrame({'MASSIF_PRA': df_coord.index.values}, index=study_index)
        df['ALTITUDE'] = self.altitude
        df_coord.index = study_index
        df_obs.index = study_index
        df_study = pd.concat([df, df_coord, df_obs], axis=1)
        return df_study

    @cached_property
    def df_merged_intersection_clean(self):
        df_stations = self.load_main_df_station_intersection_clean()
        df_study = self.load_main_df_study_intersection_clean()
        diff = set(df_study.columns).symmetric_difference(set(df_stations.columns))
        assert not diff, diff
        df = pd.concat([df_study.loc[:, df_stations.columns], df_stations], axis=0)
        df = df.sort_values([MASSIF_COLUMN_NAME])
        return df

    ##################### COMPARE THE TWO DATASETS BY FITTING THE SAME MODEL ############################

    def spatial_comparison(self, margin_model_class, default_covariance_function=CovarianceFunction.powexp):
        # max_stable_models = load_test_max_stable_models(default_covariance_function=CovarianceFunction.powexp)
        # max_stable_models = [max_stable_models[1], max_stable_models[-2]]
        max_stable_models = load_test_max_stable_models(default_covariance_function=default_covariance_function)
        max_stable_models = [m for m in max_stable_models if isinstance(m, (BrownResnick, ExtremalT))]
        for max_stable_model in max_stable_models:
            print('\n\n', get_display_name_from_object_type(type(max_stable_model)))
            if hasattr(max_stable_model, 'covariance_function'):
                print(max_stable_model.covariance_function)
            estimators = []
            datasets = [self.station_dataset, self.study_dataset_lambert]  # type: List[AbstractDataset]
            # Checks that the dataset have the same index
            assert pd.Index.equals(datasets[0].observations.columns, datasets[1].observations.columns)
            # assert datasets[0].observations.columns
            for dataset in datasets:
                margin_model = margin_model_class(coordinates=dataset.coordinates)
                estimator = FullEstimatorInASingleStepWithSmoothMargin(dataset=dataset,
                                                                       margin_model=margin_model,
                                                                       max_stable_model=max_stable_model)
                estimator.fit()
                print(estimator.result_from_model_fit.margin_coef_ordered_dict)
                print(estimator.result_from_model_fit.other_coef_dict)
                estimators.append(estimator)
            # Compare the sign of them margin coefficient for the estimators
            coefs = [{k: v for k, v in e.result_from_model_fit.margin_coef_ordered_dict.items() if 'Coeff1' not in k} for e in
                     estimators]
            different_sign = [k for k, v in coefs[0].items() if np.sign(coefs[1][k]) != np.sign(v)]
            print('All linear coefficient have the same sign: {}, different_signs for: {}'.format(
                len(different_sign) == 0, different_sign))
