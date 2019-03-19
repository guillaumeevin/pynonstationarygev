from experiment.simulation.abstract_simulation import AbstractSimulation
from extreme_estimator.estimator.full_estimator.full_estimator_for_simulation import FULL_ESTIMATORS_FOR_SIMULATION
from extreme_estimator.estimator.margin_estimator.margin_estimator_for_simulation import \
    MARGIN_ESTIMATORS_FOR_SIMULATION
from extreme_estimator.extreme_models.margin_model.linear_margin_model import ConstantMarginModel
from extreme_estimator.extreme_models.max_stable_model.max_stable_models import Smith
from extreme_estimator.margin_fits.gev.gev_params import GevParams
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import LinSpaceSpatialCoordinates
from spatio_temporal_dataset.dataset.simulation_dataset import FullSimulatedDataset


class LinSpace5Simulation(AbstractSimulation):
    FITTED_ESTIMATORS = []

    def __init__(self, nb_fit=1):
        super().__init__(nb_fit)
        # Simulation parameters
        # Number of observations
        self.nb_obs = 60
        # 1 dimensional spatial coordinates (separated in train split and test split)
        self.coordinates = LinSpaceSpatialCoordinates.from_nb_points(nb_points=100,
                                                                     train_split_ratio=0.75)
        # MarginModel Constant for simulation
        params_sample = {
            (GevParams.LOC, 0): 1.0,
            (GevParams.SHAPE, 0): 1.0,
            (GevParams.SCALE, 0): 1.0,
        }
        self.margin_model = ConstantMarginModel(coordinates=self.coordinates,
                                                params_sample=params_sample)
        # MaxStable Model for simulation
        self.max_stable_model = Smith()

    def dump(self):
        dataset = FullSimulatedDataset.from_double_sampling(nb_obs=self.nb_obs, margin_model=self.margin_model,
                                                            coordinates=self.coordinates,
                                                            max_stable_model=self.max_stable_model)
        self._dump(dataset=dataset)


if __name__ == '__main__':
    simu = LinSpace5Simulation(nb_fit=10)
    simu.dump()
    estimators_class = MARGIN_ESTIMATORS_FOR_SIMULATION + FULL_ESTIMATORS_FOR_SIMULATION
    # for estimator_class in estimators_class[:]:
    #     simu.fit(estimator_class, show=False)
    simu.visualize_comparison_graph()
