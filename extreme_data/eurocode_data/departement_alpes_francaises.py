import re
from extreme_data.eurocode_data.eurocode_region import AbstractEurocodeRegion, E, C2, C1
from root_utils import get_display_name_from_object_type


class AbstractDepartementAlpesFrancaises(object):

    def __init__(self, eurocode_region: type):
        self.eurocode_region = eurocode_region()  # type: AbstractEurocodeRegion

    def display_limit(self, ax):
        pass

    def __str__(self):
        departement_name = re.findall('[A-Z][^A-Z]*', get_display_name_from_object_type(type(self)))
        departement_name = ' '.join(departement_name)
        return  departement_name + ' ({} Region)'.format(get_display_name_from_object_type(type(self.eurocode_region)))


class HauteSavoie(AbstractDepartementAlpesFrancaises):

    def __init__(self):
        super().__init__(E)


class Savoie(AbstractDepartementAlpesFrancaises):

    def __init__(self):
        super().__init__(E)


class Isere(AbstractDepartementAlpesFrancaises):

    def __init__(self):
        super().__init__(C2)


class Drome(AbstractDepartementAlpesFrancaises):

    def __init__(self):
        super().__init__(C2)


class HautesAlpes(AbstractDepartementAlpesFrancaises):

    def __init__(self):
        super().__init__(C1)


class AlpesMaritimes(AbstractDepartementAlpesFrancaises):

    def __init__(self):
        super().__init__(C1)


class AlpesDeHauteProvence(AbstractDepartementAlpesFrancaises):

    def __init__(self):
        super().__init__(C1)
