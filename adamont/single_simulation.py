from cached_property import cached_property
from netCDF4._netCDF4 import Dataset


class SingleSimulation(object):

    def __init__(self, nc_path):
        self.nc_path = nc_path

    @cached_property
    def dataset(self):
        return Dataset(self.nc_path)



    def massif_name_and_altitude_to_return_level(self):
        return {}


