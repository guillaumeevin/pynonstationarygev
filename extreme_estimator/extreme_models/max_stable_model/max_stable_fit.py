# test how to call the max stable method
import pandas as pd

import numpy as np

from extreme_estimator.extreme_models.utils import R
from extreme_estimator.gev.gev_mle_fit import GevMleFit
import rpy2.robjects.numpy2ri as rpyn

import rpy2.robjects as robjects


def max_stable_fit():
    r = R().r
    r("""
    set.seed(42)
    n.obs = 50
    n.site = 2
    coord <- matrix(rnorm(2*n.site, sd = sqrt(.2)), ncol = 2)
    colnames(coord) = c("E", "N")

    #  Generate the data
    data <- rmaxstab(n.obs, coord, "gauss", cov11 = 100, cov12 = 25, cov22 = 220)


    loc.form = loc ~ 1
    scale.form = scale ~ 1
    shape.form = shape ~ 1
    
    namedlist = list(cov11 = 1.0, cov12 = 1.2, cov22 = 2.2, locCoeff1=1.0, scaleCoeff1=1.0, shapeCoeff1=1.0)
    # res = fitmaxstab(data=data, coord=coord, cov.mod="gauss", start=namedlist, fit.marge=TRUE, loc.form=loc.form, scale.form=scale.form,shape.form=shape.form)
    """)

    # coord = np.array(r['coord'])
    data = r['data']
    coord = pd.DataFrame(r['coord'])
    coord.colnames = robjects.StrVector(['E', 'N'])

    print(r['loc.form'])
    print(type(r['loc.form']))
    # x_gev = rpyn.numpy2ri(x_gev)
    print(coord)

    print(coord.colnames)

    # res = r.fitmaxstab(data=data, coord=coord, covmod="gauss", start=namedlist, fit.marge=TRUE, loc.form=loc.form, scale.form=scale.form,shape.form=shape.form)

    # m2.colnames = R.StrVector(['x', 'y'])
    # res = r.fitmaxstab()
    # mle_params = dict(r.attr(res, 'coef').items())
    # print(mle_params)


if __name__ == '__main__':
    max_stable_fit()
