import matplotlib.pyplot as plt

from collections import OrderedDict

from cached_property import cached_property

from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_full_name, \
    gcm_rcm_couple_to_str, get_color_from_gcm_rcm_couple
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    SCM_STUDY_CLASS_TO_ABBREVIATION
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer


class AdamontStudies(object):

    def __init__(self, study_class, gcm_rcm_couples=None, **kwargs_study):
        self.study_class = study_class
        if gcm_rcm_couples is None:
            gcm_rcm_couples = list(gcm_rcm_couple_to_full_name.keys())
        self.gcm_rcm_couples = gcm_rcm_couples
        self.gcm_rcm_couple_to_study = OrderedDict()  # type: OrderedDict[int, AbstractStudy]
        for gcm_rcm_couple in self.gcm_rcm_couples:
            study = study_class(gcm_rcm_couple=gcm_rcm_couple, **kwargs_study)
            self.gcm_rcm_couple_to_study[gcm_rcm_couple] = study

    @property
    def study_list(self):
        return list(self.gcm_rcm_couple_to_study.values())

    @cached_property
    def study(self) -> AbstractStudy:
        return self.study_list[0]

    # Some plots

    def show_or_save_to_file(self, plot_name, show=False, no_title=False, tight_layout=None):
        study_visualizer = StudyVisualizer(study=self.study, show=show, save_to_file=not show)
        study_visualizer.plot_name = plot_name
        study_visualizer.show_or_save_to_file(add_classic_title=False, dpi=500, no_title=no_title,
                                              tight_layout=tight_layout)

    def plot_maxima_time_series(self, massif_names=None, scm_study=None):
        massif_names = massif_names if massif_names is not None else self.study.all_massif_names()
        for massif_names in massif_names:
            self._plot_maxima_time_series(massif_names, scm_study)

    def _plot_maxima_time_series(self, massif_name, scm_study=None):
        ax = plt.gca()
        x = self.study.ordered_years
        linewidth = 2
        for gcm_rcm_couple, study in list(self.gcm_rcm_couple_to_study.items())[::-1]:
            if massif_name in study.massif_name_to_annual_maxima:
                y = study.massif_name_to_annual_maxima[massif_name]
                label = gcm_rcm_couple_to_str(gcm_rcm_couple)
                color = get_color_from_gcm_rcm_couple(gcm_rcm_couple)
                ax.plot(x, y, linewidth=linewidth, label=label, color=color)
        if scm_study is not None:
            try:
                print(type(scm_study))
                y = scm_study.massif_name_to_annual_maxima[massif_name]
                label = 'Reanalysis'
                color = 'black'
                ax.plot(x, y, linewidth=linewidth * 2, label=label, color=color)
            except KeyError:
                pass

        ax.xaxis.set_ticks(x[1::10])
        ax.tick_params(axis='both', which='major', labelsize=13)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1])
        plot_name = 'Annual maxima of {} in {}'.format(SCM_STUDY_CLASS_TO_ABBREVIATION[self.study_class],
                                                       massif_name.replace('_', ' '))
        ax.set_ylabel('{} ({})'.format(plot_name, self.study.variable_unit), fontsize=15)
        ax.set_xlabel('years', fontsize=15)
        plot_name = 'time series/' + plot_name
        self.show_or_save_to_file(plot_name=plot_name, show=False, no_title=True, tight_layout=True)
        ax.clear()
        plt.close()
