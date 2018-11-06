from extreme_estimator.R_fit.max_stable_fit.abstract_max_stable_model import AbstractMaxStableModel
from extreme_estimator.R_fit.max_stable_fit.max_stable_models import Smith, BrownResnick
from extreme_estimator.estimator.msp_estimator import MaxStableEstimator
from extreme_estimator.robustness_plot.abstract_robustness_plot import DisplayItem
from extreme_estimator.robustness_plot.single_scalar_plot import SingleScalarPlot
from spatio_temporal_dataset.dataset.simulation_dataset import SimulatedDataset
from spatio_temporal_dataset.spatial_coordinates.alps_station_coordinates import AlpsStationCoordinates, \
    AlpsStationCoordinatesBetweenZeroAndOne
from spatio_temporal_dataset.spatial_coordinates.generated_coordinates import CircleCoordinates


class MspSpatial(SingleScalarPlot):
    MaxStableModelItem = DisplayItem('max_stable_model', Smith)
    SpatialCoordinateClassItem = DisplayItem('spatial_coordinate_class', CircleCoordinates)
    SpatialParamsItem = DisplayItem('spatial_params', {"r": 1})
    NbStationItem = DisplayItem('nb_station', 50)
    NbObservationItem = DisplayItem('nb_obs', 60)

    def single_scalar_from_all_params(self, **kwargs_single_point) -> float:
        # Get the argument from kwargs
        max_stable_model = self.MaxStableModelItem.value_from_kwargs(**kwargs_single_point)  # type: AbstractMaxStableModel
        spatial_coordinate_class = self.SpatialCoordinateClassItem.value_from_kwargs(**kwargs_single_point)
        nb_station = self.NbStationItem.value_from_kwargs(**kwargs_single_point)
        spatial_params = self.SpatialParamsItem.value_from_kwargs(**kwargs_single_point)
        nb_obs = self.NbObservationItem.value_from_kwargs(**kwargs_single_point)
        # Run the estimation
        spatial_coordinate = spatial_coordinate_class.from_nb_points(nb_points=nb_station, **spatial_params)
        dataset = SimulatedDataset.from_max_stable_sampling(nb_obs=nb_obs, max_stable_model=max_stable_model,
                                                            spatial_coordinates=spatial_coordinate)
        estimator = MaxStableEstimator(dataset, max_stable_model)
        estimator.timed_fit()
        errors = estimator.error(max_stable_model.params_sample)
        mae_error = errors[MaxStableEstimator.MAE_ERROR]
        return mae_error


def spatial_robustness_alps():
    spatial_robustness = MspSpatial(grid_row_item=MspSpatial.NbObservationItem,
                                    grid_column_item=MspSpatial.SpatialCoordinateClassItem,
                                    plot_row_item=MspSpatial.NbStationItem,
                                    plot_label_item=MspSpatial.MaxStableModelItem)
    # Put only the parameter that will vary
    spatial_robustness.robustness_grid_plot(**{
        MspSpatial.NbStationItem.argument_name: [10, 30, 50, 70, 86][:],
        MspSpatial.NbObservationItem.argument_name: [10],
        MspSpatial.MaxStableModelItem.argument_name: [Smith(), BrownResnick()][:],
        MspSpatial.SpatialCoordinateClassItem.argument_name: [CircleCoordinates, AlpsStationCoordinatesBetweenZeroAndOne][:],
    })


if __name__ == '__main__':
    spatial_robustness_alps()
