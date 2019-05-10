import unittest
import pandas as pd
from collections import Counter, OrderedDict

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
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
from spatio_temporal_dataset.coordinates.utils import get_index_with_spatio_temporal_index_suffix
from spatio_temporal_dataset.slicer.spatio_temporal_slicer import SpatioTemporalSlicer


class TestSpatialCoordinates(unittest.TestCase):
    DISPLAY = False

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.coord = None  # type:  AbstractCoordinates

    def tearDown(self):
        if self.DISPLAY:
            self.coord.visualize()
        self.assertTrue(True)

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


if __name__ == '__main__':
    unittest.main()
