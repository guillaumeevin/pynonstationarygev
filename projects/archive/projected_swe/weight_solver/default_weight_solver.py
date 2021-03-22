from projects.archive.projected_swe.weight_solver.abtract_weight_solver import AbstractWeightSolver


class EqualWeight(AbstractWeightSolver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def couple_to_weight(self):
        couples = list(self.couple_to_historical_study.keys())
        weight = 1 / len(couples)
        return {c: weight for c in couples}
