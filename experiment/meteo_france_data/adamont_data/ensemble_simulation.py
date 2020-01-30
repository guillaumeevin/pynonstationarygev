import os
from typing import List
import os.path as op

import numpy as np
from cached_property import cached_property

from experiment.meteo_france_data.adamont_data.single_simulation import SingleSimulation

ADAMONT_PATH = r"/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/ADAMONT"


class EnsembleSimulation(object):

    def __init__(self, scenario='HISTO', parameter='SNOWSWE',
                 first_winter_required_for_histo=1958, last_winter_for_histo=2004):
        self.scenario = scenario
        self.parameter = parameter
        self.first_winter_required_for_histo = first_winter_required_for_histo
        self.last_year_for_histo = last_winter_for_histo

        # Assert value for the parameter
        assert scenario in ['HISTO', 'RCP45']
        assert parameter in ['SNOWSWE']
        assert first_winter_required_for_histo >= 1950
        assert first_winter_required_for_histo <= 2004

        # Load simulations
        # todo: so far i am using one ensemble member
        self.simulations = [SingleSimulation(nc_path, self.parameter,
                                             self.first_winter_required_for_histo,
                                             self.last_year_for_histo) for nc_path in self.nc_paths][:1]

    @cached_property
    def simulations_path(self):
        return op.join(ADAMONT_PATH, self.parameter, self.scenario)

    @cached_property
    def nc_paths(self):
        return [op.join(ADAMONT_PATH, self.parameter, self.scenario, nc_file) for nc_file in self.nc_files]

    @cached_property
    def nc_files(self) -> List[str]:
        nc_files = []
        for file in os.listdir(self.simulations_path):
            first_year = int(file.split('_')[-3][:4])
            if first_year <= self.first_winter_required_for_histo:
                # Also remove the historical simulations that contain "CNRM-CM5"
                # Problem reported in "limitations" on their website
                # Ce problème affecte toutes lessimulations HISTORIQUE CORDEX
                # réalisées en utilisant le forçage CNRM-CM5: CCLM4-8-17: ALADIN53 et RCA4
                if self.scenario == 'HISTO' and 'CNRM-CM5' in file:
                    print('here', file)
                    continue
                nc_files.append(file)
        assert len(nc_files) > 0
        return nc_files

    @cached_property
    def simulations_names(self):
        return [' + '.join(file.split('_')[2:-5]) for file in self.nc_files]

    def massif_name_and_altitude_to_mean_return_level(self):
        return {}

    @property
    def first_simulation(self):
        return self.simulations[0]

    @property
    def massif_name_and_altitude(self):
        pass

    @cached_property
    def massif_name_and_altitude_to_mean_average_annual_maxima(self):
        d = {}
        for m, a in self.first_simulation.massif_name_and_altitude_to_average_maxima.keys():
            d[(m, a)] = np.mean([s.massif_name_and_altitude_to_average_maxima[(m, a)] for s in self.simulations])
        return d


if __name__ == '__main__':
    # np.array(d.variables['SNOWSWE'])
    ensemble = EnsembleSimulation(first_winter_required_for_histo=1958)
    print(len(ensemble.simulations))
    print(ensemble.simulations_names)
    s = ensemble.first_simulation
    d = s.dataset
    # print(s.massif_name_and_altitude_to_annual_maxima_time_series)
    # print(s.massif_name_and_altitude_to_average_maxima)
    print(ensemble.massif_name_and_altitude_to_mean_average_annual_maxima)
    print(s.years)
    # print(d)
