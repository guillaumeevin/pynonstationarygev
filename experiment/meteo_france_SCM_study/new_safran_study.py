import numpy as np

from experiment.meteo_france_SCM_study.abstract_study import AbstractStudy
import os.path as op

from experiment.meteo_france_SCM_study.safran.safran_variable import SafranSnowfallVariable


class NewStudy(AbstractStudy):


    @property
    def safran_full_path(self) -> str:
        return op.join(self.full_path, 'alp_flat/reanalysis/meteo')
        # return op.join(self.full_path, 'alp_flat/reanalysis/pro')

if __name__ == '__main__':
    study = NewStudy(SafranSnowfallVariable)
    d = study.year_to_dataset_ordered_dict[1958]
    print(d)
    s = study.year_to_daily_time_serie_array[1958].shape
    print(s)
    print(s[1] / 23)
    print(d.variables['massif'])
    print(np.array(d.variables['massif']))

    for item in ['LAT', 'LON', 'ZS', 'massif']:
        a = np.array(d.variables[item])
        print(a)
        s = set(a)
        print(len(s))
