from extreme_estimator.estimator.max_stable_estimator import MaxStableEstimator
from extreme_estimator.extreme_models.max_stable_model.abstract_max_stable_model import AbstractMaxStableModel
from extreme_estimator.extreme_models.max_stable_model.max_stable_models import Smith
from experiment.robustness_plot.display_item import DisplayItem
from experiment.robustness_plot.multiple_plot import MultiplePlot
from experiment.robustness_plot.single_plot import SinglePlot
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.generated_spatial_coordinates import CircleCoordinates
from spatio_temporal_dataset.dataset.simulation_dataset import MaxStableDataset


class MaxStableDisplayItem(DisplayItem):

    def display_name_from_value(self, value: AbstractMaxStableModel):
        return value.cov_mod


class CoordinateDisplayItem(DisplayItem):

    def display_name_from_value(self, value: AbstractCoordinates):
        return str(value).split('.')[-1].split("'")[0]


class MaxStableProcessPlot(object):
    MaxStableModelItem = MaxStableDisplayItem('max_stable_model', Smith)
    CoordinateClassItem = CoordinateDisplayItem('coordinate_class', CircleCoordinates)
    NbStationItem = DisplayItem('Number of stations', 50)
    NbObservationItem = DisplayItem('nb_obs', 60)

    def msp_spatial_ordinates(self, **kwargs_single_point) -> dict:
        # Get the argument from kwargs
        max_stable_model = self.MaxStableModelItem.value_from_kwargs(
            **kwargs_single_point)  # type: AbstractMaxStableModel
        coordinate_class = self.CoordinateClassItem.value_from_kwargs(**kwargs_single_point)
        nb_station = self.NbStationItem.value_from_kwargs(**kwargs_single_point)
        nb_obs = self.NbObservationItem.value_from_kwargs(**kwargs_single_point)
        # Run the estimation
        spatial_coordinates = coordinate_class.from_nb_points(nb_points=nb_station)
        dataset = MaxStableDataset.from_sampling(nb_obs=nb_obs, max_stable_model=max_stable_model,
                                                 coordinates=spatial_coordinates)
        estimator = MaxStableEstimator(dataset, max_stable_model)
        estimator.fit()
        return estimator.scalars(max_stable_model.params_sample)


class SingleMaxStableProcessPlot(SinglePlot, MaxStableProcessPlot):

    def compute_value_from_kwargs_single_point(self, **kwargs_single_point):
        return self.msp_spatial_ordinates(**kwargs_single_point)


class MultipleMaxStableProcessPlot(MultiplePlot, MaxStableProcessPlot):

    def compute_value_from_kwargs_single_point(self, **kwargs_single_point):
        return self.msp_spatial_ordinates(**kwargs_single_point)