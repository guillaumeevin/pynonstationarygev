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


