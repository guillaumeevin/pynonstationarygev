import unittest

import numpy as np

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall, Safran
from extreme_data.meteo_france_data.scm_models_data.safran.safran_variable import SafranTemperatureVariable


class TestSafranVariables(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        study = SafranSnowfall(year_max=1960)
        self.dataset = study.year_to_dataset_ordered_dict[1959]

    def test_variables(self):
        for variable_class in Safran.SAFRAN_VARIABLES:
            keywords = variable_class.keyword()
            names = keywords if isinstance(keywords, list) else [keywords]
            variable_arrays = [np.array(self.dataset.variables[name]) for name in names]
            variable_class(*variable_arrays)
        self.assertTrue(True)




