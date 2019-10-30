from experiment.eurocode_data.utils import LAST_YEAR_FOR_EUROCODE


class AbstractEurocodeRegion(object):

    def __init__(self, sk0, sad) -> None:
        # Valeurs caracteristique de la charge de neige sur le sol à une altitude inférieure à 200m
        self.sk0 = sk0
        # Valeur de calcul de la charge exceptionelle
        self.sad = sad

    def eurocode_max_loading(self, altitude):
        valeur_caracteritique = self.valeur_caracteristique(altitude)
        if self.sad is None:
            return valeur_caracteritique
        else:
            return max(self.sad, valeur_caracteritique)

    def valeur_caracteristique(self, altitude):
        return self.sk0 + self.lois_de_variation_de_la_valeur_caracteristique(altitude)

    def lois_de_variation_de_la_valeur_caracteristique(self, altitude):
        if 200 <= altitude <= 2000:
            if 200 <= altitude <= 500:
                a, b = self.lois_de_variation_200_and_500
            elif 500 <= altitude <= 1000:
                a, b = self.lois_de_variation_500_and_1000
            else:
                a, b = self.lois_de_variation_1000_and_2000
            return a * altitude / 1000 + b
        else:
            raise ValueError('altitude {}m is out of range'.format(altitude))

    @property
    def lois_de_variation_200_and_500(self):
        return 1.0, -0.20

    @property
    def lois_de_variation_500_and_1000(self):
        return 1.5, -0.45

    @property
    def lois_de_variation_1000_and_2000(self):
        return 3.5, -2.45

    def plot_max_loading(self, ax, altitudes):
        # old_label = 'Eurocode computed in {}'.format(LAST_YEAR_FOR_EUROCODE)
        new_label = 'Eurocode standards'
        ax.plot(altitudes, [self.eurocode_max_loading(altitude) for altitude in altitudes],
                label=new_label, color='k')


class C1(AbstractEurocodeRegion):

    def __init__(self) -> None:
        super().__init__(0.65, None)


class C2(AbstractEurocodeRegion):

    def __init__(self) -> None:
        super().__init__(0.65, 1.35)


class E(AbstractEurocodeRegion):

    def __init__(self) -> None:
        super().__init__(1.40, None)

    @property
    def lois_de_variation_200_and_500(self):
        return 1.5, -0.30

    @property
    def lois_de_variation_500_and_1000(self):
        return 3.5, -1.30

    @property
    def lois_de_variation_1000_and_2000(self):
        return 7, -4.80
