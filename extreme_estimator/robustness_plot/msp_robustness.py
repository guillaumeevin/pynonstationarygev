from extreme_estimator.R_model.max_stable_model.abstract_max_stable_model import AbstractMaxStableModel, CovarianceFunction
from extreme_estimator.R_model.max_stable_model.max_stable_models import Smith, BrownResnick, Schlather, ExtremalT
from extreme_estimator.estimator.abstract_estimator import AbstractEstimator
from extreme_estimator.estimator.max_stable_estimator import MaxStableEstimator
from extreme_estimator.robustness_plot.multiple_plot import MultiplePlot
from extreme_estimator.robustness_plot.single_plot import SinglePlot
from spatio_temporal_dataset.dataset.simulation_dataset import SimulatedDataset
from spatio_temporal_dataset.spatial_coordinates.abstract_spatial_coordinates import AbstractSpatialCoordinates
from spatio_temporal_dataset.spatial_coordinates.alps_station_2D_coordinates import \
    AlpsStation2DCoordinatesBetweenZeroAndOne, AlpsStationCoordinatesBetweenZeroAndTwo
from spatio_temporal_dataset.spatial_coordinates.generated_coordinates import CircleCoordinatesRadius1, \
    CircleCoordinatesRadius2
from extreme_estimator.robustness_plot.display_item import DisplayItem


class MaxStableDisplayItem(DisplayItem):

    def display_name_from_value(self, value: AbstractMaxStableModel):
        return value.cov_mod


class SpatialCoordinateDisplayItem(DisplayItem):

    def display_name_from_value(self, value: AbstractSpatialCoordinates):
        return str(value).split('.')[-1].split("'")[0]


class MspSpatial(object):
    MaxStableModelItem = MaxStableDisplayItem('max_stable_model', Smith)
    SpatialCoordinateClassItem = SpatialCoordinateDisplayItem('spatial_coordinate_class', CircleCoordinatesRadius1)
    NbStationItem = DisplayItem('Number of stations', 50)
    NbObservationItem = DisplayItem('nb_obs', 60)

    def msp_spatial_ordinates(self, **kwargs_single_point) -> dict:
        # Get the argument from kwargs
        max_stable_model = self.MaxStableModelItem.value_from_kwargs(
            **kwargs_single_point)  # type: AbstractMaxStableModel
        spatial_coordinate_class = self.SpatialCoordinateClassItem.value_from_kwargs(**kwargs_single_point)
        nb_station = self.NbStationItem.value_from_kwargs(**kwargs_single_point)
        nb_obs = self.NbObservationItem.value_from_kwargs(**kwargs_single_point)
        # Run the estimation
        spatial_coordinate = spatial_coordinate_class.from_nb_points(nb_points=nb_station)
        dataset = SimulatedDataset.from_max_stable_sampling(nb_obs=nb_obs, max_stable_model=max_stable_model,
                                                            spatial_coordinates=spatial_coordinate)
        estimator = MaxStableEstimator(dataset, max_stable_model)
        estimator.fit()
        return estimator.scalars(max_stable_model.params_sample)


class SingleMspSpatial(SinglePlot, MspSpatial):

    def compute_value_from_kwargs_single_point(self, **kwargs_single_point):
        return self.msp_spatial_ordinates(**kwargs_single_point)


class MultipleMspSpatial(MultiplePlot, MspSpatial):

    def compute_value_from_kwargs_single_point(self, **kwargs_single_point):
        return self.msp_spatial_ordinates(**kwargs_single_point)


# def single_spatial_robustness_alps():
#     spatial_robustness = SingleMspSpatial(grid_row_item=SingleMspSpatial.NbObservationItem,
#                                           grid_column_item=SingleMspSpatial.SpatialCoordinateClassItem,
#                                           plot_row_item=SingleMspSpatial.NbStationItem,
#                                           plot_label_item=SingleMspSpatial.MaxStableModelItem)
#     # Put only the parameter that will vary
#     spatial_robustness.robustness_grid_plot(**{
#         SingleMspSpatial.NbStationItem.name: list(range(43, 87, 15)),
#         SingleMspSpatial.NbObservationItem.name: [10],
#         SingleMspSpatial.MaxStableModelItem.name: [Smith(), BrownResnick()][:],
#         SingleMspSpatial.SpatialCoordinateClassItem.name: [CircleCoordinatesRadius1,
#                                                            AlpsStationCoordinatesBetweenZeroAndOne][:],
#     })


def multiple_spatial_robustness_alps():
    nb_observation = 60
    nb_sample = 10
    plot_name = 'fast_result'
    nb_stations = list(range(43, 87, 15))
    # nb_stations = [10, 20, 30]

    spatial_robustness = MultipleMspSpatial(
        grid_column_item=MspSpatial.SpatialCoordinateClassItem,
        plot_row_item=MspSpatial.NbStationItem,
        plot_label_item=MspSpatial.MaxStableModelItem,
        nb_samples=nb_sample,
        main_title="Max stable analysis with {} years of observations".format(nb_observation),
        plot_png_filename=plot_name
        )
    # Load all the models
    msp_models = [Smith(), BrownResnick()]
    # for covariance_function in CovarianceFunction:
    #     msp_models.extend([ExtremalT(covariance_function=covariance_function)])

    # Put only the parameter that will vary
    spatial_robustness.robustness_grid_plot(**{
        SinglePlot.OrdinateItem.name: [AbstractEstimator.MAE_ERROR, AbstractEstimator.DURATION],
        MspSpatial.NbStationItem.name: nb_stations,
        MspSpatial.NbObservationItem.name: nb_observation,
        MspSpatial.MaxStableModelItem.name: msp_models,
        MspSpatial.SpatialCoordinateClassItem.name: [CircleCoordinatesRadius1,
                                                     CircleCoordinatesRadius2,
                                                     AlpsStation2DCoordinatesBetweenZeroAndOne,
                                                     AlpsStationCoordinatesBetweenZeroAndTwo][:],
    })


if __name__ == '__main__':
    # single_spatial_robustness_alps()
    multiple_spatial_robustness_alps()
