import matplotlib.pyplot as plt

from collections import OrderedDict

import numpy as np
from cached_property import cached_property

from extreme_data.meteo_france_data.adamont_data.abstract_adamont_study import AbstractAdamontStudy
from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import get_gcm_rcm_couple_adamont_to_full_name, \
    gcm_rcm_couple_to_color
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_str, \
    get_year_min_and_year_max_from_scenario
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    SCM_STUDY_CLASS_TO_ABBREVIATION, ADAMONT_STUDY_CLASS_TO_ABBREVIATION
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer


class AdamontStudies(object):

    def __init__(self, study_class, year_min_studies=None, year_max_studies=None,
                 gcm_rcm_couples=None, adamont_version=2, **kwargs_study):
        self.study_class = study_class
        self.year_min_studies = year_min_studies
        self.year_max_studies = year_max_studies
        if gcm_rcm_couples is None:
            gcm_rcm_couple_to_full_name = get_gcm_rcm_couple_adamont_to_full_name(adamont_version)
            gcm_rcm_couples = list(gcm_rcm_couple_to_full_name.keys())
        self.gcm_rcm_couples = gcm_rcm_couples
        self.gcm_rcm_couple_to_study = OrderedDict()  # type: OrderedDict[int, AbstractAdamontStudy]
        for gcm_rcm_couple in self.gcm_rcm_couples:
            study = study_class(gcm_rcm_couple=gcm_rcm_couple, adamont_version=adamont_version,
                                year_min=year_min_studies, year_max=year_max_studies, **kwargs_study)
            self.gcm_rcm_couple_to_study[gcm_rcm_couple] = study

    @property
    def study_list(self):
        return list(self.gcm_rcm_couple_to_study.values())

    @cached_property
    def study(self) -> AbstractAdamontStudy:
        return self.study_list[0]

    @property
    def nb_ensemble_members(self):
        return len(self.gcm_rcm_couples)

    # Some plots

    def show_or_save_to_file(self, plot_name, show=False, no_title=False, tight_layout=None):
        study_visualizer = StudyVisualizer(study=self.study, show=show, save_to_file=not show)
        study_visualizer.plot_name = plot_name
        study_visualizer.show_or_save_to_file(add_classic_title=False, dpi=500, no_title=no_title,
                                              tight_layout=tight_layout)

    def plot_maxima_time_series_adamont(self, massif_names=None, scm_study=None, legend_and_labels=True):
        massif_names = massif_names if massif_names is not None else self.study.all_massif_names()
        for massif_names in massif_names:
            self._plot_maxima_time_series(massif_names, scm_study, legend_and_labels=legend_and_labels)

    def _plot_maxima_time_series(self, massif_name, scm_study=None, legend_and_labels=True):
        ax = plt.gca()
        linewidth = 2
        for gcm_rcm_couple, study in list(self.gcm_rcm_couple_to_study.items())[::-1]:
            if massif_name in study.massif_name_to_annual_maxima:
                x = study.ordered_years
                y = study.massif_name_to_annual_maxima[massif_name]
                label = gcm_rcm_couple_to_str(gcm_rcm_couple)
                color = gcm_rcm_couple_to_color[gcm_rcm_couple]
                ax.plot(x, y, linewidth=linewidth, label=label, color=color)
        # if scm_study is None:
        #     pass
            # I should recode that, taking into account that the length of annual maxima is not the same
            # for all the time series
            # x = study.ordered_years
            # y = np.array([study.massif_name_to_annual_maxima[massif_name] for study in self.study_list
            #      if massif_name in study.massif_name_to_annual_maxima])
            # if len(y) > 0:
            #     y = np.mean(y, axis=0)
            #     label = 'Mean maxima'
            #     color = 'black'
            #     ax.plot(x, y, linewidth=linewidth * 2, label=label, color=color)
        # else:
            # todo: otherwise display the mean in strong black
            # try:
            #     x = scm_study.ordered_years
            #     y = scm_study.massif_name_to_annual_maxima[massif_name]
            #     label = 'Reanalysis'
            #     color = 'black'
            #     ax.plot(x, y, linewidth=linewidth * 2, label=label, color=color)
            # except KeyError:
            #     pass

        ticks = [year for year in range(self.year_min_studies, self.year_max_studies+1) if year % 10 == 0]
        ax.xaxis.set_ticks(ticks)
        ax.yaxis.grid()
        ax.set_xlim((self.year_min_studies, self.year_max_studies))
        if legend_and_labels:
            # Augment the ylim for the legend
            ylim_min, ylim_max = ax.get_ylim()
            ax.set_ylim((ylim_min, ylim_max * 1.5))
            ax.tick_params(axis='both', which='major', labelsize=13)
            handles, labels = ax.get_legend_handles_labels()
            ncol = 2 if self.study.adamont_version == 1 else 3
            ax.legend(handles[::-1], labels[::-1], ncol=ncol, prop={'size': 7})
        plot_name = 'Annual maxima of {} in {} at {} m'.format(ADAMONT_STUDY_CLASS_TO_ABBREVIATION[self.study_class],
                                                       massif_name.replace('_', ' '),
                                                        self.study.altitude)
        fontsize = 13
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        if legend_and_labels:
            ax.set_ylabel('{} ({})'.format(plot_name, self.study.variable_unit), fontsize=fontsize)
            ax.set_xlabel('years', fontsize=fontsize)
        plot_name = 'time series/' + plot_name
        self.show_or_save_to_file(plot_name=plot_name, show=False, no_title=True, tight_layout=True)
        ax.clear()
        plt.close()
