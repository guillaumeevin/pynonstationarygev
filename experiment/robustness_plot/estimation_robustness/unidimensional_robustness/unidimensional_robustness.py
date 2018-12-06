from extreme_estimator.estimator.abstract_estimator import AbstractEstimator
from extreme_estimator.extreme_models.max_stable_model.max_stable_models import Smith, BrownResnick
from experiment.robustness_plot.estimation_robustness.max_stable_process_plot import MultipleMaxStableProcessPlot, MaxStableProcessPlot
from experiment.robustness_plot.single_plot import SinglePlot
from spatio_temporal_dataset.coordinates.spatial_coordinates.alps_station_2D_coordinates import \
    AlpsStation2DCoordinatesBetweenZeroAndOne, AlpsStationCoordinatesBetweenZeroAndTwo
from spatio_temporal_dataset.coordinates.spatial_coordinates.generated_spatial_coordinates import CircleSpatialCoordinates, \
    CircleSpatialCoordinatesRadius2


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


def multiple_unidimensional_robustness():
    nb_observation = 20
    nb_sample = 1
    plot_name = 'fast_result'
    nb_stations = list(range(43, 87, 15))
    # nb_stations = [10, 20, 30]

    spatial_robustness = MultipleMaxStableProcessPlot(
        grid_column_item=MaxStableProcessPlot.CoordinateClassItem,
        plot_row_item=MaxStableProcessPlot.NbStationItem,
        plot_label_item=MaxStableProcessPlot.MaxStableModelItem,
        nb_samples=nb_sample,
        main_title="Max stable analysis with {} years of spatio_temporal_observations".format(nb_observation),
        plot_png_filename=plot_name
    )
    # Load all the models
    msp_models = [Smith(), BrownResnick()]
    # for covariance_function in CovarianceFunction:
    #     msp_models.extend([ExtremalT(covariance_function=covariance_function)])

    # Put only the parameter that will vary
    spatial_robustness.robustness_grid_plot(**{
        SinglePlot.OrdinateItem.name: [AbstractEstimator.MAE_ERROR, AbstractEstimator.DURATION],
        MaxStableProcessPlot.NbStationItem.name: nb_stations,
        MaxStableProcessPlot.NbObservationItem.name: nb_observation,
        MaxStableProcessPlot.MaxStableModelItem.name: msp_models,
        MaxStableProcessPlot.CoordinateClassItem.name: [CircleSpatialCoordinates,
                                                        CircleSpatialCoordinatesRadius2,
                                                        AlpsStation2DCoordinatesBetweenZeroAndOne,
                                                        AlpsStationCoordinatesBetweenZeroAndTwo][:],
    })


if __name__ == '__main__':
    # single_spatial_robustness_alps()
    multiple_unidimensional_robustness()
