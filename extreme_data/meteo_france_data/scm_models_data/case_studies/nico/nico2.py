import pandas as pd
import numpy as np
import xlsxwriter


from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusDepth, CrocusSwe3Days
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranSnowfall3Days
from extreme_data.meteo_france_data.scm_models_data.utils import FrenchRegion
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    SCM_STUDY_CLASS_TO_ABBREVIATION
from extreme_fit.estimator.margin_estimator.utils import fitted_stationary_gev
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies


def generate_excel_with_annual_maxima(fast=True, maxima_dates=False):
    if fast:
        altitudes = [600, 900]
    else:
        altitudes = [600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]
    study_class = SafranSnowfall1Day
    study_name = 'annual maxima of ' + SCM_STUDY_CLASS_TO_ABBREVIATION[study_class]
    if maxima_dates:
        study_name += ' - number of days since 1st August, e.g. 1 represents the 2nd of August'
    writer = pd.ExcelWriter('{}.xlsx'.format(study_name), engine='xlsxwriter')
    altitude_studies = AltitudesStudies(study_class, altitudes)
    for massif_name in altitude_studies.study.all_massif_names():
        write_df_with_annual_maxima(massif_name, writer, altitude_studies, maxima_dates)
    writer.save()


def write_df_with_annual_maxima(massif_name, writer, altitude_studies, maxima_dates=False) -> pd.DataFrame:
    columns = []
    altitudes = []
    for altitude, study in altitude_studies.altitude_to_study.items():
        df_maxima = study.observations_annual_maxima.df_maxima_gev
        if massif_name in study.study_massif_names:
            altitudes.append(altitude)
            s = df_maxima.loc[massif_name]
            if maxima_dates:
                values = study.massif_name_to_annual_maxima_index[massif_name]
                s = pd.Series(index=s.index, data=values)
                # s.values = np.array(values)
            # Fit the data and add the parameters as the first columns
            columns.append(s)
    df = pd.concat(columns, axis=1)
    altitude_str = [str(a) + ' m' for a in altitudes]
    df.columns = altitude_str
    df.to_excel(writer, sheet_name=massif_name)
    return df


if __name__ == '__main__':
    generate_excel_with_annual_maxima(fast=False, maxima_dates=True)
