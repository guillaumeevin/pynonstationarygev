from typing import Dict, Tuple

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from projects.projected_swe.weight_solver.abtract_weight_solver import AbstractWeightSolver


class EqualWeight(AbstractWeightSolver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def couple_to_weight(self):
        couples = list(self.couple_to_historical_study.keys())
        weight = 1 / len(couples)
        return {c: weight for c in couples}
