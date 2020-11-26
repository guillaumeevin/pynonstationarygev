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
        altitudes = [900, 1200]
    else:
        altitudes = [900, 1200, 1500, 1800, 2100]
    study_class = SafranSnowfall1Day
    study_name = 'annual maxima of ' + SCM_STUDY_CLASS_TO_ABBREVIATION[study_class]
    if maxima_dates:
        study_name += ' - number of days since 1st August, e.g. 1 represents the 2nd of August'
    writer = pd.ExcelWriter('{}.xlsx'.format(study_name), engine='xlsxwriter')
    altitude_studies = AltitudesStudies(study_class, altitudes)
    for altitude, study in altitude_studies.altitude_to_study.items():
        write_df_with_annual_maxima_v2(altitude, writer, study, maxima_dates)
    writer.save()


def write_df_with_annual_maxima_v2(altitude, writer, study, maxima_dates=False) -> pd.DataFrame:
    df = study.df_latitude_longitude
    data_list = []
    for massif_name in df.index:
        if maxima_dates:
            values = study.massif_name_to_annual_maxima_angle[massif_name]
        else:
            raise NotImplementedError
        data_list.append(values)
    data = np.array(data_list)
    df2 = pd.DataFrame(data=data, index=df.index, columns=study.ordered_years).astype(float)
    df = pd.concat([df, df2], axis=1)
    print(df.head())
    df.to_excel(writer, sheet_name='altitude = {} m'.format(altitude))

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
