from experiment.split.split_curve import SplitCurve, LocFunction
from extreme_estimator.estimator.full_estimator import FullEstimatorInASingleStepWithSmoothMargin
from extreme_estimator.extreme_models.margin_model.smooth_margin_model import ConstantMarginModel
from extreme_estimator.extreme_models.max_stable_model.max_stable_models import Smith
from extreme_estimator.gev_params import GevParams
from spatio_temporal_dataset.coordinates.unidimensional_coordinates.coordinates_1D import LinSpaceCoordinates

from spatio_temporal_dataset.dataset.simulation_dataset import FullSimulatedDataset
from spatio_temporal_dataset.slicer.spatial_slicer import SpatialSlicer
from spatio_temporal_dataset.slicer.spatio_temporal_slicer import SpatioTemporalSlicer


def load_dataset():
    nb_points = 50
    nb_obs = 60
    coordinates = LinSpaceCoordinates.from_nb_points(nb_points=nb_points, train_split_ratio=0.8)

    # MarginModel Linear with respect to the shape (from 0.01 to 0.02)
    params_sample = {
        # (GevParams.GEV_SHAPE, 0): 0.2,
        (GevParams.GEV_LOC, 0): 10,
        (GevParams.GEV_SHAPE, 0): 1.0,
        (GevParams.GEV_SCALE, 0): 1.0,
    }
    margin_model = ConstantMarginModel(coordinates=coordinates, params_sample=params_sample)
    max_stable_model = Smith()

    return FullSimulatedDataset.from_double_sampling(nb_obs=nb_obs, margin_model=margin_model,
                                                     coordinates=coordinates,
                                                     max_stable_model=max_stable_model,
                                                     train_split_ratio=0.8,
                                                     slicer_class=SpatioTemporalSlicer)


def full_estimator(dataset):
    max_stable_model = Smith()
    margin_model_for_estimator = ConstantMarginModel(dataset.coordinates)
    full_estimator = FullEstimatorInASingleStepWithSmoothMargin(dataset, margin_model_for_estimator, max_stable_model)
    return full_estimator


if __name__ == '__main__':
    dataset = load_dataset()
    dataset.slicer.summary()
    full_estimator = full_estimator(dataset)
    curve = SplitCurve(dataset, full_estimator, margin_functions=[LocFunction()])
    curve.visualize()
