import numpy as np
from cached_property import cached_property
import matplotlib.pyplot as plt

from extreme_data.meteo_france_data.scm_models_data.cluster.clustering_total_precip import TotalPrecipCluster
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.estimator.margin_estimator.utils import fitted_stationary_gev
from extreme_fit.model.margin_model.utils import MarginFitMethod
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies


class PointWIseGevAnalysisForCluster(AltitudesStudies):

    def __init__(self, study_class, altitudes,
                 spatial_transformation_class=None, temporal_transformation_class=None,
                 cluster_class=TotalPrecipCluster,
                 **kwargs_study):
        super().__init__(study_class, altitudes, spatial_transformation_class, temporal_transformation_class,
                         **kwargs_study)
        self.cluster_class = cluster_class

    # Plot for the Altitude

    def cluster_id_to_annual_maxima_list(self, first_year=1959, last_year=2019, altitudes=None):
        if altitudes is None:
            altitudes = self.altitudes
        cluster_id_to_annual_maxima_list = {}
        for cluster_id in self.cluster_class.cluster_ids:
            annual_maxima_list = []
            massif_names = self.cluster_class.cluster_id_to_massif_names[cluster_id]
            for study in [s for a, s in self.altitude_to_study.items() if a in altitudes]:
                annual_maxima = []
                massif_ids = [study.massif_name_to_massif_id[m] for m in massif_names if
                              m in study.massif_name_to_massif_id]
                for year in range(first_year, last_year + 1):
                    annual_maxima.extend(study.year_to_annual_maxima[year][massif_ids])
                annual_maxima_list.append(annual_maxima)
            cluster_id_to_annual_maxima_list[cluster_id] = annual_maxima_list
        return cluster_id_to_annual_maxima_list

    @property
    def cluster_id_to_gev_params_list(self):
        cluster_id_to_gev_params_list = {}
        for cluster_id, annual_maxima_list in self.cluster_id_to_annual_maxima_list().items():
            gev_params_list = []
            for annual_maxima in annual_maxima_list:
                if len(annual_maxima) > 0:
                    gev_params = fitted_stationary_gev(annual_maxima, fit_method=MarginFitMethod.extremes_fevd_mle)
                else:
                    gev_params = GevParams(0, 0, 0)
                    assert gev_params.has_undefined_parameters
                gev_params_list.append(gev_params)
            cluster_id_to_gev_params_list[cluster_id] = gev_params_list
        return cluster_id_to_gev_params_list

    def plot_gev_parameters_against_altitude(self):
        for param_name in GevParams.PARAM_NAMES[:]:
            ax = plt.gca()
            for cluster_id, gev_param_list in self.cluster_id_to_gev_params_list.items():
                params, altitudes = zip(*[(gev_param.to_dict()[param_name], altitude) for gev_param, altitude
                                          in zip(gev_param_list, self.altitudes) if
                                          not gev_param.has_undefined_parameters])
                # ax.plot(self.altitudes, params, label=str(cluster_id), linestyle='None', marker='x')
                label = self.cluster_class.cluster_id_to_cluster_name[cluster_id]
                ax.plot(altitudes, params, label=label, marker='x')
            ax.legend()
            ax.set_xlabel('Altitude')
            ax.set_ylabel(param_name)
            plot_name = '{} change with altitude'.format(param_name)
            self.show_or_save_to_file(plot_name, no_title=True, tight_layout=True, show=False)
            ax.clear()
            plt.close()

    # Plot for the time

    @property
    def year_min_and_max_list(self):
        l = []
        year_min, year_max = 1959, 1989
        for shift in range(0, 7):
            l.append((year_min + 5 * shift, year_max + 5 * shift))
        return l

    def cluster_id_to_time_annual_maxima_list(self, altitudes):
        cluster_id_to_time_annual_maxima_list = {cluster_id: [] for cluster_id in self.cluster_class.cluster_ids}
        for year_min, year_max in self.year_min_and_max_list:
            d = self.cluster_id_to_annual_maxima_list(first_year=year_min,
                                                      last_year=year_max,
                                                      altitudes=altitudes)
            for cluster_id, annual_maxima_list in d.items():
                a = np.array(annual_maxima_list)
                a = a.flatten()
                cluster_id_to_time_annual_maxima_list[cluster_id].append(list(a))
        return cluster_id_to_time_annual_maxima_list

    def cluster_id_to_time_gev_params_list(self, altitudes):
        cluster_id_to_gev_params_list = {}
        for cluster_id, annual_maxima_list in self.cluster_id_to_time_annual_maxima_list(altitudes=altitudes).items():
            print("cluster", cluster_id)
            gev_params_list = []
            for annual_maxima in annual_maxima_list:
                if len(annual_maxima) > 20:
                    print(type(annual_maxima))
                    annual_maxima = np.array(annual_maxima)
                    print(annual_maxima.shape)
                    annual_maxima = annual_maxima[annual_maxima != 0]
                    gev_params = fitted_stationary_gev(annual_maxima)
                else:
                    print('here all the time')
                    gev_params = GevParams(0, 0, 0)
                    assert gev_params.has_undefined_parameters
                gev_params_list.append(gev_params)
            cluster_id_to_gev_params_list[cluster_id] = gev_params_list
        return cluster_id_to_gev_params_list

    def plot_gev_parameters_against_time(self, altitudes=None):
        for param_name in GevParams.PARAM_NAMES[:]:
            ax = plt.gca()
            for cluster_id, gev_param_list in self.cluster_id_to_time_gev_params_list(altitudes).items():
                print(gev_param_list)
                params, years = zip(*[(gev_param.to_dict()[param_name], years) for gev_param, years
                                      in zip(gev_param_list, self.year_min_and_max_list) if
                                      not gev_param.has_undefined_parameters])
                # ax.plot(self.altitudes, params, label=str(cluster_id), linestyle='None', marker='x')
                label = self.cluster_class.cluster_id_to_cluster_name[cluster_id]
                years = [year[0] for year in years]
                ax.plot(years, params, label=label, marker='x')
            ax.legend()
            ax.set_xlabel('Year')
            ax.set_ylabel(param_name)
            xlabels = ['-'.join([str(e) for e in t]) for t in self.year_min_and_max_list]
            ax.set_xticklabels(xlabels)
            plot_name = '{} change with year for altitudes {}'.format(param_name, '+'.join([str(a) for a in altitudes]))
            self.show_or_save_to_file(plot_name, no_title=True, tight_layout=True, show=False)
            ax.clear()
            plt.close()


if __name__ == '__main__':
    altitudes = [600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]
    # altitudes = [1800, 2100]
    pointwise_gev_analysis = PointWIseGevAnalysisForCluster(SafranSnowfall1Day, altitudes=altitudes)
    # pointwise_gev_analysis.plot_gev_parameters_against_time(altitudes)

    # pointwise_gev_analysis.plot_gev_parameters_against_altitude()
    for altitudes in [[600, 900],
                      [1200, 1500, 1800],
                      [2100, 2400, 2700],
                      [3000, 3300, 3600]][3:]:
        print(altitudes)
        pointwise_gev_analysis.plot_gev_parameters_against_time(altitudes)
