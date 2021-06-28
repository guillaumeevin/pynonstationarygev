import numpy as np

from extreme_trend.ensemble_simulation.simulation_generator_with_effect.simulation_versions_v3 import \
    AbstractSimulationForSnowLoadAt1500


class AbstractMeanExperiment(AbstractSimulationForSnowLoadAt1500):

    @property
    def average_bias_reference(self):
        return self.ray_simulation, 0

    @property
    def ray_simulation(self):
        raise NotImplementedError

class MeanExperiment5(AbstractMeanExperiment):

    @property
    def ray_simulation(self):
        return 5

class MeanExperiment7_5(AbstractMeanExperiment):

    @property
    def ray_simulation(self):
        return 7.5


class MeanExperiment10(AbstractMeanExperiment):

    @property
    def ray_simulation(self):
        return 10


class MeanExperiment12_5(AbstractMeanExperiment):

    @property
    def ray_simulation(self):
        return 12.5
