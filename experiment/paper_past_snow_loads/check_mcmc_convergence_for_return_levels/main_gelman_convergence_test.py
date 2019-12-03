import pandas as pd

from experiment.paper_past_snow_loads.paper_main_utils import load_altitude_to_visualizer
from experiment.paper_past_snow_loads.paper_utils import paper_altitudes, paper_study_classes
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel
from root_utils import get_display_name_from_object_type


def gelman_convergence_test(mcmc_iterations, model_class, altitudes, study_class, nb_chains=3, massif_names=None):
    """ Test if a given number of MCMC iterations is enough for the convergence of each parameter of the model_class
     for every time series with non-null values present in the study_classes

     Ideally, return a DataFrame with altitude as columns, and massif name as index that contain the R score
     Then we could compute the max R that should ideally be below 1.2
     """

    altitude_to_visualizer = load_altitude_to_visualizer(altitudes, massif_names=massif_names,
                                                         non_stationary_uncertainty=None,
                                                         study_class=study_class, uncertainty_methods=None)
    altitude_to_d = {}
    for altitude, vizu in altitude_to_visualizer.items():
        altitude_to_d[altitude] = vizu.massif_name_to_gelman_convergence_value(mcmc_iterations, model_class, nb_chains)
    massif_names = list(altitude_to_d[altitudes[0]].keys())
    df = pd.DataFrame(altitude_to_d, index=massif_names, columns=altitudes)
    return df



"""
test gelman for the 4 types of models
and the for the 3 variables considered: GSL, GSL from eurocode, GLS in 3 days
"""

if __name__ == '__main__':
    for half_mcmc_iterations in [500000][:]:
        for study_class in paper_study_classes[:1]:
            study_name = get_display_name_from_object_type(study_class)
            print(study_name, half_mcmc_iterations)
            # df = gelman_convergence_test(mcmc_iterations=mcmc_iterations, altitudes=paper_altitudes[:],
            #                              study_class=study_class, model_class=StationaryTemporalModel,
            #                              massif_names=None,
            #                              nb_chains=3)
            mcmc_iterations = 2 * half_mcmc_iterations
            df = gelman_convergence_test(mcmc_iterations=mcmc_iterations, altitudes=paper_altitudes[:1],
                                         study_class=study_class, model_class=StationaryTemporalModel,
                                         massif_names=['Vercors'],
                                         nb_chains=3)
            csv_filename = '{}_{}_{}_{}.csv'.format(study_name, mcmc_iterations, df.max().max(), df.mean().mean())
            df.to_csv(csv_filename)


            #
            # print(df.head())
            # df.to_csv(csv_filename)
            # print('Overall maxima for {} iterations:'.format(mcmc_iterations), df.max().max())
