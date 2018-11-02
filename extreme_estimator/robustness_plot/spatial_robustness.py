from extreme_estimator.R_fit.max_stable_fit.max_stable_models import GaussianMSP, BrownResick
from extreme_estimator.robustness_plot.abstract_robustness import DisplayItem, AbstractRobustnessPlot, \
    SpatialCoordinateClassItem, NbObservationItem, NbStationItem, MaxStableModelItem, SpatialParamsItem
from spatio_temporal_dataset.spatial_coordinates.generated_coordinate import CircleCoordinates

spatial_robustness = AbstractRobustnessPlot(grid_row_item=SpatialCoordinateClassItem,
                                            grid_column_item=NbObservationItem,
                                            plot_row_item=NbStationItem,
                                            plot_label_item=MaxStableModelItem)

# Put only the parameter that will vary
spatial_robustness.robustness_grid_plot(**{
    NbStationItem.argument_name: [10, 20, 30, 40, 50],
    MaxStableModelItem.argument_name: [GaussianMSP(), BrownResick()][:]
})
