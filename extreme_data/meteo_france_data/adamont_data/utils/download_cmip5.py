import cdsapi

c = cdsapi.Client()

models = ['mpi_esm_lr', 'cnrm_cm5']


def get_year_min_and_max(model, experiment):
    pass


def retrieve(model, experiment):
    year_min, year_max = get_year_min_and_max(model, experiment)
    c.retrieve(
        'projections-cmip5-monthly-single-levels',
        {
            'experiment': 'historical',
            'variable': '2m_temperature',
            'model': 'cnrm_cm5',
            'ensemble_member': 'r1i1p1',
            'period': '195001-200512',
            'format': 'zip',
        },
        'download.zip')
