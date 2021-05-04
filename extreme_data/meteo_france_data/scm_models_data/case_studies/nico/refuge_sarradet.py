import pandas as pd
import numpy as np
import xlsxwriter


from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusDepth, CrocusSwe3Days
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranSnowfall3Days
from extreme_data.meteo_france_data.scm_models_data.utils import FrenchRegion
from extreme_fit.estimator.margin_estimator.utils import fitted_stationary_gev


def generate_excel_with_annual_maxima(fit=True):
    massif_name, french_region, altitudes = 'Beaufortain', FrenchRegion.alps, [900, 1200]
    # massif_name, french_region, altitudes = 'haute-bigorre', FrenchRegion.pyrenees, [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300]
    study_classes = [SafranSnowfall1Day, SafranSnowfall3Days, CrocusSwe3Days, CrocusDepth]
    writer = pd.ExcelWriter('{}.xlsx'.format(massif_name), engine='xlsxwriter')
    # writer = pd.ExcelWriter('pandas_multiple.xlsx')
    for j, study_class in enumerate(study_classes, 1):
        write_df_with_annual_maxima(massif_name, french_region, writer, study_class, altitudes, fit=fit)
    writer.save()


def write_df_with_annual_maxima(massif_name, french_region, writer, study_class, altitudes, fit=True) -> pd.DataFrame:
    columns = []
    for altitude in altitudes:
        study = study_class(altitude=altitude, french_region=french_region) # type: AbstractStudy
        df_maxima = study.observations_annual_maxima.df_maxima_gev
        s = df_maxima.loc[massif_name]
        # Fit the data and add the parameters as the first columns
        gev_param = fitted_stationary_gev(s.values)
        s = pd.concat([gev_param.to_serie(), s])
        columns.append(s)
    df = pd.concat(columns, axis=1)
    altitude_str = [str(a) + ' m' for a in altitudes]
    df.columns = altitude_str
    name = study.variable_name
    short_name = name[:31]
    df.to_excel(writer, scheet_name=short_name)
    return df


if __name__ == '__main__':
    generate_excel_with_annual_maxima(fit=True)
