import unittest

import numpy as np

from extreme_estimator.extreme_models.margin_model.smooth_margin_model import ConstantMarginModel, \
    LinearAllParametersAllDimsMarginModel
from extreme_estimator.extreme_models.utils import r
from extreme_estimator.margin_fits.gev.gev_params import GevParams
from spatio_temporal_dataset.dataset.simulation_dataset import FullSimulatedDataset
from test.test_unitary.test_rmaxstab.test_rmaxstab_without_margin import TestRMaxStab
from test.test_unitary.test_unitary_abstract import TestUnitaryAbstract


class TestRMaxStabWithMarginConstant(TestUnitaryAbstract):

    @classmethod
    def r_code(cls):
        TestRMaxStab.r_code()
        r("""
        param.shape <- rep(0.2, n.site)
        param.loc <- rep(0.2, n.site)
        param.scale <- rep(0.2, n.site)""")
        r("""
        for (i in 1:n.site)
        data[,i] <- frech2gev(data[,i], param.loc[i], param.scale[i], param.shape[i])
        """)


    @classmethod
    def python_code(cls):
        coordinates, max_stable_model = TestRMaxStab.python_code()
        # Load margin model
        gev_param_name_to_coef_list = {
            GevParams.LOC: [0.2],
            GevParams.SHAPE: [0.2],
            GevParams.SCALE: [0.2],
        }
        margin_model = ConstantMarginModel.from_coef_list(coordinates, gev_param_name_to_coef_list)
        # Load dataset
        dataset = FullSimulatedDataset.from_double_sampling(nb_obs=40, margin_model=margin_model,
                                                            coordinates=coordinates,
                                                            max_stable_model=max_stable_model)

        return dataset

    @property
    def r_output(self):
        self.r_code()
        return np.sum(r.data)

    @property
    def python_output(self):
        dataset = self.python_code()
        return np.sum(dataset.maxima_gev())

    def test_rmaxstab_with_constant_margin(self):
        self.compare()


class TestRMaxStabWithLinearMargin(TestUnitaryAbstract):

    @classmethod
    def r_code(cls):
        TestRMaxStab.r_code()
        r("""
        param.loc <- -10 + 2 * locations[,2]
        ##TODO-IMPLEMENT SQUARE: param.scale <- 5 + 2 * locations[,1] + locations[,2]^2
        param.shape <- 2 -3 * locations[,1]
        param.scale <- 5 + 2 * locations[,1] + locations[,2]
        ##Transform the unit Frechet margins to GEV
        for (i in 1:n.site)
        data[,i] <- frech2gev(data[,i], param.loc[i], param.scale[i], param.shape[i])
        """)

    @classmethod
    def python_code(cls):
        coordinates, max_stable_model = TestRMaxStab.python_code()
        # Load margin model
        gev_param_name_to_coef_list = {
            GevParams.LOC: [-10, 0, 2],
            GevParams.SHAPE: [2, -3, 0],
            GevParams.SCALE: [5, 2, 1],
        }
        margin_model = LinearAllParametersAllDimsMarginModel.from_coef_list(coordinates, gev_param_name_to_coef_list)
        # Load dataset
        dataset = FullSimulatedDataset.from_double_sampling(nb_obs=40, margin_model=margin_model,
                                                            coordinates=coordinates,
                                                            max_stable_model=max_stable_model)
        return dataset

    @property
    def r_output(self):
        self.r_code()
        return np.sum(r.data)

    @property
    def python_output(self):
        dataset = self.python_code()
        return np.sum(dataset.maxima_gev())

    def test_rmaxstab_with_linear_margin(self):
        self.compare()


if __name__ == '__main__':
    unittest.main()
