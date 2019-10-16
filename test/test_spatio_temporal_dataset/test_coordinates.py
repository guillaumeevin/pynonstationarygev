import unittest
import numpy as np
import pandas as pd
from collections import Counter, OrderedDict

from extreme_fit.model.utils import set_seed_for_test
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.abstract_spatio_temporal_coordinates import \
    AbstractSpatioTemporalCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.generated_spatio_temporal_coordinates import \
    UniformSpatioTemporalCoordinates, GeneratedSpatioTemporalCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import UniformSpatialCoordinates, \
    LinSpaceSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.alps_station_2D_coordinates import \
    AlpsStation2DCoordinatesBetweenZeroAndOne
from spatio_temporal_dataset.coordinates.spatial_coordinates.alps_station_3D_coordinates import \
    AlpsStation3DCoordinatesWithAnisotropy
from spatio_temporal_dataset.coordinates.spatial_coordinates.generated_spatial_coordinates import \
    CircleSpatialCoordinates
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation import \
    CenteredScaledNormalization
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.uniform_normalization import \
    BetweenZeroAndOneNormalization
from spatio_temporal_dataset.coordinates.utils import get_index_with_spatio_temporal_index_suffix
from spatio_temporal_dataset.slicer.spatio_temporal_slicer import SpatioTemporalSlicer
from test.test_utils import load_test_spatiotemporal_coordinates, load_test_spatial_coordinates, \
    load_test_temporal_coordinates, load_test_1D_and_2D_spatial_coordinates


class TestSpatialCoordinates(unittest.TestCase):
    DISPLAY = False

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.coord = None  # type:  AbstractCoordinates

    def tearDown(self):
        if self.DISPLAY:
            self.coord.visualize()
        first_coordinate = self.coord.df_all_coordinates.iloc[0, 0]
        self.assertFalse(np.isnan(first_coordinate))

    def test_unif(self):
        self.coord = UniformSpatialCoordinates.from_nb_points(nb_points=10)

    def test_circle(self):
        self.coord = CircleSpatialCoordinates.from_nb_points(nb_points=500)

    def test_normalization(self):
        self.coord = AlpsStation2DCoordinatesBetweenZeroAndOne.from_csv()

    def test_anisotropy(self):
        self.coord = AlpsStation3DCoordinatesWithAnisotropy.from_csv()


class SpatioTemporalCoordinates(unittest.TestCase):
    nb_points = 4
    nb_steps = 2

    def test_temporal_circle(self):
        self.coordinates = UniformSpatioTemporalCoordinates.from_nb_points_and_nb_steps(nb_points=self.nb_points,
                                                                                        nb_steps=self.nb_steps,
                                                                                        train_split_ratio=0.5)
        c = Counter([len(self.coordinates.df_coordinates(split)) for split in SpatioTemporalSlicer.SPLITS])
        good_count = c == Counter([2, 2, 2, 2]) or c == Counter([0, 0, 4, 4])
        self.assertTrue(good_count)

    def test_unique_spatio_temporal_index_and_matching_spatial_index(self):
        spatial_coordinates = LinSpaceSpatialCoordinates.from_nb_points(self.nb_points)
        spatial_indexes = [[10, 11, 12, 13], ['a', 'b', 'c', 'd']]
        for spatial_index in spatial_indexes:
            spatial_coordinates.df_all_coordinates.index = spatial_index
            df_spatial = spatial_coordinates.df_spatial_coordinates()
            coordinates = GeneratedSpatioTemporalCoordinates.from_df_spatial_and_nb_steps(df_spatial=df_spatial,
                                                                                          nb_steps=self.nb_steps)

            # the uniqueness of each spatio temporal index is not garanteed by the current algo
            # it will work in classical cases, and raise an assert when uniqueness is needed (when using a slicer)
            index1 = pd.Series(spatial_coordinates.spatial_index())
            index2 = pd.Series(coordinates.spatial_index())
            ind = index1 != index2  # type: pd.Series
            self.assertEqual(sum(ind), 0, msg="spatial_coordinates:\n{} \n!= spatio_temporal_coordinates \n{}".
                             format(index1.loc[ind], index2.loc[ind]))

            index1 = get_index_with_spatio_temporal_index_suffix(spatial_coordinates.df_spatial_coordinates(), t=0)
            index1 = pd.Series(index1)
            index2 = pd.Series(coordinates.df_spatial_coordinates().index)
            ind = index1 != index2  # type: pd.Series
            self.assertEqual(sum(ind), 0, msg="spatial_coordinates:\n{} \n!= spatio_temporal_coordinates \n{}".
                             format(index1.loc[ind], index2.loc[ind]))

    def test_ordered_coordinates(self):
        # Order coordinates, to ensure that the first dimension/the second dimension and so on..
        # Always are in the same order to a given type (e.g. spatio_temporal= of coordinates
        # Check space coordinates
        d = OrderedDict()
        d[AbstractCoordinates.COORDINATE_Z] = [1]
        d[AbstractCoordinates.COORDINATE_X] = [1]
        d[AbstractCoordinates.COORDINATE_Y] = [1]
        df = pd.DataFrame.from_dict(d)
        for df2 in [df, df.loc[:, ::-1]][-1:]:
            coordinates = AbstractCoordinates(df=df2, slicer_class=SpatioTemporalSlicer)
            self.assertEqual(list(coordinates.df_all_coordinates.columns),
                             [AbstractCoordinates.COORDINATE_X, AbstractCoordinates.COORDINATE_Y,
                              AbstractCoordinates.COORDINATE_Z])
        # Check space/time ordering
        d = OrderedDict()
        d[AbstractCoordinates.COORDINATE_T] = [1]
        d[AbstractCoordinates.COORDINATE_X] = [1]
        df = pd.DataFrame.from_dict(d)
        for df2 in [df, df.loc[:, ::-1]][-1:]:
            coordinates = AbstractCoordinates(df=df2, slicer_class=SpatioTemporalSlicer)
            self.assertEqual(list(coordinates.df_all_coordinates.columns),
                             [AbstractCoordinates.COORDINATE_X, AbstractCoordinates.COORDINATE_T])


class TestCoordinatesWithTransformedStartingPoint(unittest.TestCase):

    def setUp(self) -> None:
        set_seed_for_test(seed=42)
        self.nb_points = 2
        self.nb_steps = 50
        self.nb_obs = 1

    def test_starting_point_with_zero_one_normalization(self):
        # Load some 2D spatial coordinates
        coordinates = load_test_spatiotemporal_coordinates(nb_steps=self.nb_steps, nb_points=self.nb_points,
                                                           transformation_class=BetweenZeroAndOneNormalization)[
            1]  # type: AbstractSpatioTemporalCoordinates
        df = coordinates.df_temporal_coordinates_for_fit(starting_point=2)
        start_coordinates = df.iloc[2, 0]
        self.assertEqual(start_coordinates, 0.0)

    def test_starting_point_with_centered_scaled_normalization(self):
        # Load some 2D spatial coordinates
        spatial_coordinate = load_test_1D_and_2D_spatial_coordinates(nb_points=self.nb_points,
                                                                     transformation_class=BetweenZeroAndOneNormalization)[
            0]
        temporal_coordinates = \
        load_test_temporal_coordinates(nb_steps=self.nb_steps, transformation_class=CenteredScaledNormalization)[0]
        coordinates = AbstractSpatioTemporalCoordinates.from_spatial_coordinates_and_temporal_coordinates(
            spatial_coordinates=spatial_coordinate,
            temporal_coordinates=temporal_coordinates)
        # Check that df_all_coordinates have not yet been normalized
        self.assertEqual(coordinates.df_temporal_coordinates(transformed=False).iloc[-1, 0], 49.0)
        # Check that the normalization is working
        self.assertAlmostEqual(coordinates.df_temporal_coordinates_for_fit(starting_point=None).iloc[0, 0], -1.697749375254331)
        self.assertAlmostEqual(coordinates.df_temporal_coordinates_for_fit(starting_point=2).iloc[2, 0], -1.5739459974625107)
        self.assertNotEqual(coordinates.df_temporal_coordinates_for_fit(starting_point=2).iloc[2, 0],
                            coordinates.df_temporal_coordinates_for_fit(starting_point=2).iloc[3, 0])


if __name__ == '__main__':
    unittest.main()
