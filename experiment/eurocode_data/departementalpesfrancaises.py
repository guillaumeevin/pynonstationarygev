from enum import Enum

from experiment.eurocode_data.region_eurocode import AbstractRegionType, E, C2, C1


class AbstractDepartementAlpesFrancaises(object):

    def __init__(self, region: type):
        self.region = region()  # type: AbstractRegionType


class HauteSavoie(AbstractDepartementAlpesFrancaises):

    def __init__(self):
        super().__init__(E)


class Savoie(AbstractDepartementAlpesFrancaises):

    def __init__(self):
        super().__init__(E)


class Isere(AbstractDepartementAlpesFrancaises):

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


class Drome(AbstractDepartementAlpesFrancaises):

    def __init__(self):
        super().__init__(C2)


"""
Quand c'est à cheval, je mets les deux massifs
Quand juste un bout du massif est dans un autre departement 
(Chartreuse, Belledonne sont un peu en Savoie
"""
massif_name_to_departements = {
    'Chablais': [HauteSavoie],
    'Aravis': [HauteSavoie, Savoie],
    'Mont-Blanc': [HauteSavoie],
    'Bauges': [HauteSavoie, Savoie],
    'Beaufortain': [HauteSavoie, Savoie],
    'Haute-Tarentaise': [Savoie],
    'Chartreuse': [Isere],
    'Belledonne': [Isere],
    'Maurienne': [Savoie],
    'Vanoise': [Savoie],
    'Haute-Maurienne': [Savoie],
    'Grandes-Rousses': [Isere, Savoie],
    'Thabor': [HauteSavoie],
    'Vercors': [Isere, Drome],
    'Oisans': [Isere, HautesAlpes],
    'Pelvoux': [Isere, HautesAlpes],
    'Queyras': [HautesAlpes],
    'Devoluy': [Drome, Isere, HautesAlpes],
    'Champsaur': [HautesAlpes],
    'Parpaillon': [HautesAlpes, AlpesDeHauteProvence],
    'Ubaye': [AlpesDeHauteProvence],
    'Haut_Var-Haut_Verdon': [AlpesDeHauteProvence],
    'Mercantour': [AlpesMaritimes, AlpesDeHauteProvence]}
