from experiment.fit_diagnosis.split_curve import SplitCurve
from extreme_estimator.estimator.full_estimator import FullEstimatorInASingleStepWithSmoothMargin
from extreme_estimator.estimator.margin_estimator import SmoothMarginEstimator
from extreme_estimator.extreme_models.margin_model.smooth_margin_model import ConstantMarginModel, \
    LinearAllParametersAllDimsMarginModel
from extreme_estimator.extreme_models.max_stable_model.max_stable_models import Smith
from extreme_estimator.gev_params import GevParams
from spatio_temporal_dataset.coordinates.spatial_coordinates.alps_station_2D_coordinates import \
    AlpsStation2DCoordinates, AlpsStation2DCoordinatesBetweenZeroAndOne
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import LinSpaceSpatialCoordinates
from spatio_temporal_dataset.dataset.simulation_dataset import FullSimulatedDataset


class SplitCurveAlps(SplitCurve):

    def __init__(self, nb_fit: int = 1):
        super().__init__(nb_fit)
        self.nb_obs = 2
        # nb_points = len(AlpsStation2DCoordinates.from_csv())
        nb_points = 10
        self.coordinates = AlpsStation2DCoordinatesBetweenZeroAndOne.from_nb_points(nb_points, train_split_ratio=0.8)
        # MarginModel Linear with respect to the shape (from 0.01 to 0.02)
        params_sample = {
            (GevParams.GEV_LOC, 0): 10,
            (GevParams.GEV_SHAPE, 0): 1.0,
            (GevParams.GEV_SCALE, 0): 10,
        }
        self.margin_model = ConstantMarginModel(coordinates=self.coordinates, params_sample=params_sample)
        self.max_stable_model = Smith()

    def load_dataset(self):
        return FullSimulatedDataset.from_double_sampling(nb_obs=self.nb_obs, margin_model=self.margin_model,
                                                         coordinates=self.coordinates,
                                                         max_stable_model=self.max_stable_model)

    def load_estimator(self, dataset):
        max_stable_model = Smith()
        margin_model_for_estimator = LinearAllParametersAllDimsMarginModel(dataset.coordinates)
        # estimator = FullEstimatorInASingleStepWithSmoothMargin(dataset, margin_model_for_estimator, max_stable_model)
        estimator = SmoothMarginEstimator(dataset, margin_model_for_estimator)
        return estimator


if __name__ == '__main__':
    curve = SplitCurveAlps(nb_fit=2)
    curve.fit()
