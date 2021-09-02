import numpy as np
import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranPrecipitation1Day, \
    SafranTemperature
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    SCM_STUDY_CLASS_TO_ABBREVIATION
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies


def generate_excel_with_annual_maxima(fast=True):
    if fast:
        altitudes = [900]
    else:
        altitudes = [900, 1200, 1500, 1800, 2100]
    for study_class in [SafranPrecipitation1Day, SafranTemperature]:
        prefix = 'total' if study_class is SafranPrecipitation1Day else "mean"
        study_name = prefix + ' ' + SCM_STUDY_CLASS_TO_ABBREVIATION[study_class]
        writer = pd.ExcelWriter('{}.xlsx'.format(study_name), engine='xlsxwriter')
        altitude_studies = AltitudesStudies(study_class, altitudes)
        for altitude, study in altitude_studies.altitude_to_study.items():
            write_df_with_annual_maxima(altitude, writer, study)
        writer.save()


def write_df_with_annual_maxima(altitude, writer, study) -> pd.DataFrame:
    df = study.df_latitude_longitude
    data_list = []
    for massif_name in df.index:
        # values = study.massif_name_to_annual_total[massif_name]
        values = study.massif_name_to_annual_maxima[massif_name]
        data_list.append(values)
    data = np.array(data_list)
    df2 = pd.DataFrame(data=data, index=df.index, columns=study.ordered_years).astype(float)
    # df = pd.concat([df, df2], axis=1)
    df = df2
    print(df.head())
    df.to_excel(writer, sheet_name='altitude = {} m'.format(altitude))


if __name__ == '__main__':
    generate_excel_with_annual_maxima(fast=False)
