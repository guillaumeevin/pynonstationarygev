from experiment.meteo_france_SCM_study.safran.safran import Safran
from itertools import product

from experiment.meteo_france_SCM_study.safran.safran_visualizer import SafranVisualizer


def load_all_safran(only_first_one=False):
    all_safran = []
    for safran_alti, nb_day in product([1800, 2400], [1, 3, 7]):
        print('alti: {}, nb_day: {}'.format(safran_alti, nb_day))
        all_safran.append(Safran(safran_alti, nb_day))
        if only_first_one:
            break
    return all_safran


if __name__ == '__main__':
    for safran in load_all_safran(only_first_one=True):
        safran_visualizer = SafranVisualizer(safran)
        # safran_visualizer.visualize_independent_margin_fits(threshold=[None, 20, 40, 60][0])
        safran_visualizer.visualize_smooth_margin_fit()
        # safran_visualizer.visualize_full_fit()
