import unittest

import numpy as np

from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import Crocus, CrocusDepth
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall, Safran
from extreme_data.meteo_france_data.scm_models_data.safran.safran_variable import SafranTemperatureVariable


class TestVariables(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset = None
        self.variables = None

    def run_test_variables(self):
        for variable_class in self.variables:
            keywords = variable_class.keyword()
            names = keywords if isinstance(keywords, list) else [keywords]
            variable_arrays = [np.array(self.dataset.variables[name]) for name in names]
            variable_class(*variable_arrays)
        self.assertTrue(True)


class TestSafranVariables(TestVariables):

    def setUp(self) -> None:
        super().setUp()
        study = SafranSnowfall(year_max=1960)
        self.dataset = study.year_to_dataset_ordered_dict[1959]
        self.variables = Safran.SAFRAN_VARIABLES

    def test_variables(self):
        self.run_test_variables()


class TestCrocusVariables(TestVariables):

    def setUp(self) -> None:
        super().setUp()
        study = CrocusDepth(year_max=1960)
        self.dataset = study.year_to_dataset_ordered_dict[1959]
        self.variables = Crocus.CROCUS_VARIABLES

    def test_variables(self):
        self.run_test_variables()


if __name__ == '__main__':
    unittest.main()
