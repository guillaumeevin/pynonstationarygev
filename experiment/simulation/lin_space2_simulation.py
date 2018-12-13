from experiment.simulation.abstract_simulation import AbstractSimulation
from extreme_estimator.estimator.full_estimator.full_estimator_for_simulation import FULL_ESTIMATORS_FOR_SIMULATION
from extreme_estimator.estimator.margin_estimator.margin_estimator_for_simulation import \
    MARGIN_ESTIMATORS_FOR_SIMULATION
from extreme_estimator.extreme_models.margin_model.smooth_margin_model import ConstantMarginModel
from extreme_estimator.extreme_models.max_stable_model.max_stable_models import Smith
from extreme_estimator.gev_params import GevParams
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import LinSpaceSpatialCoordinates
from spatio_temporal_dataset.dataset.simulation_dataset import FullSimulatedDataset


class LinSpace3Simulation(AbstractSimulation):
    FITTED_ESTIMATORS = []

    def __init__(self, nb_fit=1):
        super().__init__(nb_fit)
        # Simulation parameters
        self.nb_obs = 60
        self.coordinates = LinSpaceSpatialCoordinates.from_nb_points(nb_points=100, train_split_ratio=0.75)
        # MarginModel Linear with respect to the shape (from 0.01 to 0.02)
        params_sample = {
            (GevParams.GEV_LOC, 0): 1.0,
            (GevParams.GEV_SHAPE, 0): 1.0,
            (GevParams.GEV_SCALE, 0): 1.0,
        }
        self.margin_model = ConstantMarginModel(coordinates=self.coordinates, params_sample=params_sample)
        self.max_stable_model = Smith()

    def dump(self):
        dataset = FullSimulatedDataset.from_double_sampling(nb_obs=self.nb_obs, margin_model=self.margin_model,
                                                            coordinates=self.coordinates,
                                                            max_stable_model=self.max_stable_model)
        self._dump(dataset=dataset)


if __name__ == '__main__':
    simu = LinSpace3Simulation(nb_fit=7)
    simu.dump()
    for estimator_class in MARGIN_ESTIMATORS_FOR_SIMULATION + FULL_ESTIMATORS_FOR_SIMULATION:
        simu.fit(estimator_class, show=False)
    simu.visualize_comparison_graph()
