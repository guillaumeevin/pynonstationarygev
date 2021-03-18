import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

from matplotlib.lines import Line2D
from scipy.special import softmax
import numpy as np

from extreme_data.meteo_france_data.adamont_data.adamont_scenario import scenario_to_str
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from projects.projected_swe.old_weight_computer.utils import save_to_filepath
from projects.projected_swe.weight_solver.abtract_weight_solver import AbstractWeightSolver
from projects.projected_swe.weight_solver.default_weight_solver import EqualWeight
from projects.projected_swe.weight_solver.indicator import AbstractIndicator, WeightComputationException
from projects.projected_swe.weight_solver.knutti_weight_solver import KnuttiWeightSolver
from projects.projected_swe.weight_solver.knutti_weight_solver_with_bootstrap import \
    KnuttiWeightSolverWithBootstrapVersion2, KnuttiWeightSolverWithBootstrapVersion1
from root_utils import get_display_name_from_object_type


class ModelAsTruth(object):

    def __init__(self, observation_study: AbstractStudy,
                 couple_to_study_projected: Dict[Tuple[str, str], AbstractStudy],
                 couple_to_study_historical: Dict[Tuple[str, str], AbstractStudy],
                 indicator_class: type,
                 knutti_weight_solver_classes,
                 massif_names=None,
                 add_interdependence_weight=False):
        self.knutti_weight_solver_classes = knutti_weight_solver_classes
        self.massif_names = massif_names
        self.add_interdependence_weight = add_interdependence_weight
        self.indicator_class = indicator_class
        self.couple_to_study_historical = couple_to_study_historical
        self.couple_to_study_projected = couple_to_study_projected
        self.observation_study = observation_study
        # Parameters
        self.width = 2
        self.solver_class_to_color = {
            EqualWeight: 'white',
            KnuttiWeightSolver: 'yellow',
            KnuttiWeightSolverWithBootstrapVersion1: 'orange',
            KnuttiWeightSolverWithBootstrapVersion2: 'red',
        }

        # Update massif names
        study_list = [observation_study] + list(couple_to_study_historical.values()) + list(couple_to_study_projected.values())
        self.massif_names = self.get_massif_names_subset_from_study_list(study_list)
        print('Nb of massifs')
        print(len(self.massif_names))
        # Set some parameters to speed up results (by caching some results)
        for study in study_list:
            study._massif_names_for_cache = self.massif_names

    def plot_against_sigma(self, sigma_list):
        ax = plt.gca()
        solver_class_to_score_list = self.get_solver_class_to_score_list(sigma_list)
        all_x = []
        labels = []
        colors = []
        for j, (solver_class, score_list) in enumerate(list(solver_class_to_score_list.items())):
            x_list = self.get_x_list(j, sigma_list)
            all_x.extend(x_list)
            assert len(x_list) == len(sigma_list)
            label = get_display_name_from_object_type(solver_class)
            color = self.solver_class_to_color[solver_class]
            print(solver_class, score_list, np.array(score_list).mean(axis=1), np.median(np.array(score_list), axis=1))
            bplot = ax.boxplot(score_list, positions=x_list, widths=self.width, patch_artist=True, showmeans=True,
                               labels=[str(sigma) for sigma in sigma_list])
            for patch in bplot['boxes']:
                patch.set_facecolor(color)
            colors.append(color)
            labels.append(label)

        custom_lines = [Line2D([0], [0], color=color, lw=4) for color in colors]
        ax.legend(custom_lines, labels, prop={'size': 8}, loc='upper left')
        ax.set_xlim(min(all_x) - self.width, max(all_x) + self.width)
        study_projected = list(self.couple_to_study_projected.values())[0]
        title = 'crpss between a weighted forecast and an unweighted forecast \n' \
                      'at {} m for {} of snowfall for {}-{} (%)'.format(self.observation_study.altitude,
                                                                        self.indicator_class.str_indicator(),
                                                                        study_projected.year_min,
                                                                        study_projected.year_max)
        ax2 = ax.twiny()
        ax2.set_xlabel('{} for {} GCM/RCM couples'.format(scenario_to_str(study_projected.scenario), len(self.couple_to_study_projected)))
        ax.set_xlabel('sigma skill parameter')
        ax.set_ylabel(title)

        # Plot a zero horizontal line
        lim_left, lim_right = ax.get_xlim()
        ax.hlines(0, xmin=lim_left, xmax=lim_right, linestyles='dashed')

        # Save or show file
        visualizer = StudyVisualizer(self.observation_study, show=False, save_to_file=True)
        visualizer.plot_name = title.split('\n')[1]
        visualizer.show_or_save_to_file(no_title=True)

        plt.close()

    def get_x_list(self, j, sigma_list):
        shift = len(self.knutti_weight_solver_classes) + 1
        x_list = [((j + 1) * (self.width * 1.1)) + shift * i * self.width for i in range(len(sigma_list))]
        return x_list

    def get_solver_class_to_score_list(self, sigma_list):
        return {solver_class: [self.compute_score(solver_class, sigma) for sigma in sigma_list]
                for solver_class in self.knutti_weight_solver_classes}

    def compute_score(self, solver_class, sigma):
        # return [sigma, sigma*2]
        score_list = []
        for gcm_rcm_couple in self.couple_to_study_historical.keys():
            historical_observation_study = self.couple_to_study_historical[gcm_rcm_couple]
            projected_observation_study = self.couple_to_study_projected[gcm_rcm_couple]
            couple_to_study_historical = {c: s for c, s in self.couple_to_study_historical.items() if
                                          c != gcm_rcm_couple}
            couple_to_study_projected = {c: s for c, s in self.couple_to_study_projected.items() if c != gcm_rcm_couple}

            try:
                if issubclass(solver_class, KnuttiWeightSolver):
                    weight_solver = solver_class(sigma, None, historical_observation_study, couple_to_study_historical,
                                                 self.indicator_class, self.massif_names, self.add_interdependence_weight,
                                                 )  # type: AbstractWeightSolver
                else:
                    weight_solver = solver_class(historical_observation_study, couple_to_study_historical,
                                                 self.indicator_class, self.massif_names, self.add_interdependence_weight,
                                                 )  # type: AbstractWeightSolver

                print(solver_class, sigma, weight_solver.couple_to_weight.values())
                mean_score = weight_solver.mean_prediction_score(self.massif_names, couple_to_study_projected,
                                                                 projected_observation_study)
                print(mean_score)
                if mean_score < 1e4:
                    score_list.append(mean_score)
            except WeightComputationException:
                pass
        # print(solver_class, sigma, score_list)
        return np.array(score_list)

    def get_massif_names_subset_from_study_list(self, study_list: List[AbstractStudy]):
        massifs_names_list = [set(s.study_massif_names) for s in study_list]
        massif_names_intersection = massifs_names_list[0].intersection(*massifs_names_list[1:])
        if self.massif_names is not None:
            massif_names_intersection = massif_names_intersection.intersection(set(self.massif_names))
        return list(massif_names_intersection)
