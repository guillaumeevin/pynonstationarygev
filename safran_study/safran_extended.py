from collections import OrderedDict

import pandas as pd

from safran_study.safran import Safran
from utils import cached_property


class ExtendedSafran(Safran):

    @property
    def safran_massif_names(self):
        return self.region_names + super().safran_massif_names

    """ Properties """

    @cached_property
    def df_annual_maxima(self):
        df_annual_maxima = pd.DataFrame(self.year_to_annual_maxima, index=super().safran_massif_names).T
        # Add the column corresponding to the aggregated massif
        for region_name_loop in self.region_names:
            # We use "is in" so that the "Alps" will automatically regroup all the massif data
            massif_belong_to_the_group = [massif_name
                                          for massif_name, region_name in self.massif_name_to_region_name.items()
                                          if region_name_loop in region_name]
            df_annual_maxima[region_name_loop] = df_annual_maxima.loc[:, massif_belong_to_the_group].max(axis=1)
        return df_annual_maxima

    @property
    def massif_name_to_region_name(self):
        df_centroid = self.load_df_centroid()
        return OrderedDict(zip(df_centroid['NOM'], df_centroid['REGION']))

    @property
    def region_names(self):
        return ['Alps', 'Northern Alps', 'Central Alps', 'Southern Alps', 'Extreme South Alps']
