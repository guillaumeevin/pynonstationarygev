from typing import List

from spatio_temporal_dataset.marginals.abstract_marginals import AbstractMarginals
from R.gev_fit.gev_marginal import GevMarginal
from spatio_temporal_dataset.stations.station import Station


class SpatialMarginal(AbstractMarginals):

    def __init__(self, stations: List[Station]):
        super().__init__(stations)
        for station in self.stations:
            self.gev_marginals.append(GevMarginal(coordinate=station, data=station.annual_maxima.values))