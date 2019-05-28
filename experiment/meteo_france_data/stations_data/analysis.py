from collections import OrderedDict
from collections import Counter

from cached_property import cached_property

from experiment.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall
from experiment.meteo_france_data.visualization.study_visualization.main_study_visualizer import \
    ALL_ALTITUDES_WITH_20_STATIONS_AT_LEAST, ALL_ALTITUDES
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.uniform_normalization import \
    BetweenZeroAndOneNormalization
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import AnnualMaxima

DATA_PATH = r'/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/Johan_data/PrecipitationsAvalanches_MaxPrecipit_ParPoste_ParHiver_traites.xls'

import pandas as pd


class Stations(object):

    def __init__(self, altitude=900, use_study_coordinate_with_latitude_and_longitude=False):
        assert altitude in [900, 1200]
        self.altitude = altitude
        self.transformation_class = BetweenZeroAndOneNormalization
        self.use_study_coordinate_with_latitude_and_longitude = use_study_coordinate_with_latitude_and_longitude
        self.year_min = 1958
        self.year_max = 2004

    ##################### STATION ATTRIBUTES ############################

    def load_main_df(self):
        df = pd.read_excel(DATA_PATH, sheet_name='max alpes 2500m presentes')
        df = df.iloc[:78, 4:]
        return df

    def load_main_df_for_altitude(self):
        df = pd.read_excel(DATA_PATH, sheet_name='max alpes 2500m presentes')
        df = df.iloc[:78]
        margin = 150
        ind_altitude = self.altitude - margin < df['ALTITUDE']
        ind_altitude &= df['ALTITUDE'] <= self.altitude + margin
        df = df.loc[ind_altitude]  # type: pd.DataFrame
        # Remove dulpicate for commune, Pellafol we should keep the first, i.e. 930 which has more data than the other
        df.drop_duplicates(subset='COMMUNE', inplace=True)
        df.set_index('COMMUNE', inplace=True)
        df = df.iloc[:, 3:]
        return df

    def load_main_df_for_altitude_and_good_massifs(self):
        df = self.load_main_df_for_altitude().copy()
        # Keep only the massif that also belong to the study (so that the normalization are roughly comparable)
        ind_massif = df['MASSIF_PRA'].isin(self.intersection_massif_names)
        df = df.loc[ind_massif]
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
        df = df.iloc[:, 7:]
        df.columns = df.columns.astype(int)
        df = df.loc[:, self.year_min:self.year_max]
        return AbstractSpatioTemporalObservations(df_maxima_gev=df)

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
        return SafranSnowfall(altitude=self.altitude, nb_consecutive_days=1, year_min=self.year_min, year_max=self.year_max+1)

    @cached_property
    def intersection_massif_names(self):
        intersection_of_massif_names = list(set(self.massif_names).intersection(set(self.study.study_massif_names)))
        diff_due_to_wrong_names = set(self.massif_names) - set(self.study.study_massif_names)
        assert not diff_due_to_wrong_names, diff_due_to_wrong_names
        return intersection_of_massif_names

    @property
    def study_coordinates(self):
        # Build coordinate, from two possibles dataframes for the coordinates
        if self.use_study_coordinate_with_latitude_and_longitude:
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
        return observations

    @property
    def study_dataset(self):
        dataset = AbstractDataset(observations=self.study_observations,
                                  coordinates=self.study_coordinates)
        return dataset

    # After a short analysis (run df_altitude to check) we decided on the altitude
    # 900 and 1200 seems to be the best altitudes

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
        # Mean number of non-Nan values
        df_values = df.iloc[:, 7:]
        df_values_from_1958 = df_values.iloc[:, 13:]
        d['Percentage of Nan'] = df_values_from_1958.isna().mean().mean()
        print(df_values_from_1958.columns)
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


if __name__ == '__main__':
    s = Stations(altitude=1200)
    print(len(s.stations_coordinates))
    print(len(s.study_coordinates))
    print(s.station_dataset)
    print(s.study_dataset)
