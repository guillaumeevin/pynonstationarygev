import numpy as np

from extreme_trend.ensemble_simulation.simulation_generator_with_effect.simulation_versions_v3 import \
    AbstractSimulationForSnowLoadAt1500


class AbstractSTDExperiment(AbstractSimulationForSnowLoadAt1500):

    @property
    def average_bias_reference(self):
        return 0, self.ray_simulation

    @property
    def ray_simulation(self):
        raise NotImplementedError

class STDExperiment5(AbstractSTDExperiment):

    @property
    def ray_simulation(self):
        return 5

class STDExperiment7_5(AbstractSTDExperiment):

    @property
    def ray_simulation(self):
        return 7.5


class STDExperiment10(AbstractSTDExperiment):

    @property
    def ray_simulation(self):
        return 10


class STDExperiment12_5(AbstractSTDExperiment):

    @property
    def ray_simulation(self):
        return 12.5
