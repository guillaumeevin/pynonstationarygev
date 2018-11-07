from extreme_estimator.R_fit.max_stable_fit.abstract_max_stable_model import AbstractMaxStableModel
from extreme_estimator.R_fit.max_stable_fit.max_stable_models import Smith, BrownResnick
from extreme_estimator.estimator.abstract_estimator import AbstractEstimator
from extreme_estimator.estimator.unitary_msp_estimator import MaxStableEstimator
from extreme_estimator.robustness_plot.multiple_plot import MultiplePlot
from extreme_estimator.robustness_plot.single_plot import SinglePlot
from spatio_temporal_dataset.dataset.simulation_dataset import SimulatedDataset
from spatio_temporal_dataset.spatial_coordinates.alps_station_coordinates import AlpsStationCoordinatesBetweenZeroAndOne
from spatio_temporal_dataset.spatial_coordinates.generated_coordinates import CircleCoordinates
from extreme_estimator.robustness_plot.display_item import DisplayItem


class MaxStableDisplayItem(DisplayItem):

    def display_name_from_value(self, value: AbstractMaxStableModel):
        return value.cov_mod


class MspSpatial(object):
    MaxStableModelItem = MaxStableDisplayItem('max_stable_model', Smith)
    SpatialCoordinateClassItem = DisplayItem('spatial_coordinate_class', CircleCoordinates)
    SpatialParamsItem = DisplayItem('spatial_params', {"r": 1})
    NbStationItem = DisplayItem('Number of stations', 50)
    NbObservationItem = DisplayItem('nb_obs', 60)

    def msp_spatial_ordinates(self, **kwargs_single_point) -> dict:
        # Get the argument from kwargs
        max_stable_model = self.MaxStableModelItem.value_from_kwargs(
            **kwargs_single_point)  # type: AbstractMaxStableModel
        spatial_coordinate_class = self.SpatialCoordinateClassItem.value_from_kwargs(**kwargs_single_point)
        nb_station = self.NbStationItem.value_from_kwargs(**kwargs_single_point)
        spatial_params = self.SpatialParamsItem.value_from_kwargs(**kwargs_single_point)
        nb_obs = self.NbObservationItem.value_from_kwargs(**kwargs_single_point)
        # Run the estimation
        spatial_coordinate = spatial_coordinate_class.from_nb_points(nb_points=nb_station, **spatial_params)
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
        print('here')
        return self.msp_spatial_ordinates(**kwargs_single_point)


def single_spatial_robustness_alps():
    spatial_robustness = SingleMspSpatial(grid_row_item=SingleMspSpatial.NbObservationItem,
                                          grid_column_item=SingleMspSpatial.SpatialCoordinateClassItem,
                                          plot_row_item=SingleMspSpatial.NbStationItem,
                                          plot_label_item=SingleMspSpatial.MaxStableModelItem)
    # Put only the parameter that will vary
    spatial_robustness.robustness_grid_plot(**{
        SingleMspSpatial.NbStationItem.name: [10, 30, 50, 70, 86][:],
        SingleMspSpatial.NbObservationItem.name: [10],
        SingleMspSpatial.MaxStableModelItem.name: [Smith(), BrownResnick()][:],
        SingleMspSpatial.SpatialCoordinateClassItem.name: [CircleCoordinates,
                                                           AlpsStationCoordinatesBetweenZeroAndOne][:],
    })


def multiple_spatial_robustness_alps():
    spatial_robustness = MultipleMspSpatial(
        grid_column_item=MspSpatial.MaxStableModelItem,
        plot_row_item=MspSpatial.NbStationItem,
        plot_label_item=MspSpatial.SpatialCoordinateClassItem,
        nb_samples=10)
    # Put only the parameter that will vary
    spatial_robustness.robustness_grid_plot(**{
        SinglePlot.OrdinateItem.name: [AbstractEstimator.MAE_ERROR, AbstractEstimator.DURATION],
        MspSpatial.NbStationItem.name: [10, 20, 30, 50, 70, 86][:3],
        MspSpatial.NbObservationItem.name: 10,
        MspSpatial.MaxStableModelItem.name: [Smith(), BrownResnick()][:],
        MspSpatial.SpatialCoordinateClassItem.name: [CircleCoordinates,
                                                     AlpsStationCoordinatesBetweenZeroAndOne][:],
    })


if __name__ == '__main__':
    # single_spatial_robustness_alps()
    multiple_spatial_robustness_alps()
