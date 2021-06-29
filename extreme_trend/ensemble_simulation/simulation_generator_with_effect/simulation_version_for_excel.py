import numpy as np

from extreme_trend.ensemble_simulation.simulation_generator_with_effect.simulation_versions_v3 import \
    AbstractSimulationForSnowLoadAt1500


class ShiftExperiment__0__0(AbstractSimulationForSnowLoadAt1500):

    @property
    def average_bias_reference(self):
        return 0, 0


class ShiftExperiment__10__0(AbstractSimulationForSnowLoadAt1500):

    @property
    def average_bias_reference(self):
        return 10, 0





class ShiftExperiment__0__10(AbstractSimulationForSnowLoadAt1500):

    @property
    def average_bias_reference(self):
        return 0, 10

class ShiftExperiment__10__10(AbstractSimulationForSnowLoadAt1500):

    @property
    def average_bias_reference(self):
        return 10, 10

class ShiftExperiment__20__0(AbstractSimulationForSnowLoadAt1500):

    @property
    def average_bias_reference(self):
        return 20, 0

class ShiftExperiment__0__20(AbstractSimulationForSnowLoadAt1500):

    @property
    def average_bias_reference(self):
        return 0, 20





class ShiftExperiment__20__20(AbstractSimulationForSnowLoadAt1500):

    @property
    def average_bias_reference(self):
        return 20, 20

class ShiftExperiment__20__10(AbstractSimulationForSnowLoadAt1500):

    @property
    def average_bias_reference(self):
        return 20, 10


class ShiftExperiment__5__0(AbstractSimulationForSnowLoadAt1500):

    @property
    def average_bias_reference(self):
        return 5, 0

class ShiftExperiment__0__5(AbstractSimulationForSnowLoadAt1500):

    @property
    def average_bias_reference(self):
        return 0, 5





class ShiftExperiment__5__5(AbstractSimulationForSnowLoadAt1500):

    @property
    def average_bias_reference(self):
        return 5, 5

class ShiftExperiment__5__10(AbstractSimulationForSnowLoadAt1500):

    @property
    def average_bias_reference(self):
        return 5, 10


class ShiftExperiment__10__5(AbstractSimulationForSnowLoadAt1500):

    @property
    def average_bias_reference(self):
        return 10, 5