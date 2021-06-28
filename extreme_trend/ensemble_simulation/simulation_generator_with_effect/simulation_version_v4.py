import numpy as np

from extreme_trend.ensemble_simulation.simulation_generator_with_effect.simulation_versions_v3 import \
    AbstractSimulationForSnowLoadAt1500


class AbstractCenteredExperiment(AbstractSimulationForSnowLoadAt1500):

    @property
    def average_bias_reference(self):
        return  self.ray_simulation / np.sqrt(2), self.ray_simulation / np.sqrt(2)

    @property
    def ray_simulation(self):
        raise NotImplementedError

class CenterExperiment7_5(AbstractCenteredExperiment):

    @property
    def ray_simulation(self):
        return 7.5


class CenterExperiment10(AbstractCenteredExperiment):

    @property
    def ray_simulation(self):
        return 10


class CenterExperiment12_5(AbstractCenteredExperiment):

    @property
    def ray_simulation(self):
        return 12.5
