from typing import List

from R.gev_fit.gev_marginal import GevMarginal, frechet_unitary_transformation
from spatio_temporal_dataset.stations.station import Station


class AbstractMarginals(object):

    def __init__(self, stations: List[Station]):
        self.stations = stations
        self.gev_marginals = []  # type: List[GevMarginal]

        #  Compute some temporal arguments
        self.years_of_study = self.stations[0].year_of_study
        self.temporal_window_size = 20


        self.smoothing_function = None

    def sample_marginals(self, smoothing_function):
        pass

    @property
    def gev_parameters(self):
        return [gev_marginal.gev_parameters_estimated for gev_marginal in self.gev_marginals]


class FrechetUnitaryTransformationFunction(object):
    pass

class NoSmoothing(object):

    def gev_parameters(self, coordinate):
        pass

class ContinuousSmoothing(object):

    def frechet_unitary_transformation(self, data, coordinate):
        gev_parameters = self.gev_parameters(coordinate)
        transformed_data = frechet_unitary_transformation(data, gev_parameters)


    def gev_parameters(self, coordinate):
        return
        return ()

if __name__ == '__main__':
    pass