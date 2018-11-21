library(SpatialExtremes)


# Boolean for python call
call_main = !exists("python_wrapping")

if (call_main) {
    set.seed(42)
    n.obs = 50
    n.site = 2
    coord <- matrix(rnorm(2*n.site, sd = sqrt(.2)), ncol = 2)
    colnames(coord) = c("E", "N")

    #  Generate the data
    data <- rmaxstab(n.obs, coord, "gauss", cov11 = 100, cov12 = 25, cov22 = 220)
    # data <- rmaxstab(n.obs, coord, "brown", range = 3, smooth = 0.5)
    # data <- rmaxstab(n.obs, coord, "whitmat", nugget = 0.0, range = 3, smooth = 0.5)
    #  Fit back the data
    # print(data)n
    # res = fitmaxstab(data, coord, "gauss", fit.marge=FALSE, )
    # res = fitmaxstab(data, coord, "brown")
    # res = fitmaxstab(data, coord, "whitmat", start=)
    print(class(coord))
    print(colnames(coord))

    loc.form = loc ~ 1
    scale.form = scale ~ 1
    shape.form = shape ~ 1


    namedlist = list(cov11 = 1.0, cov12 = 1.2, cov22 = 2.2, locCoeff1=1.0, scaleCoeff1=1.0, shapeCoeff1=1.0)
    res = fitmaxstab(data=data, coord=coord, cov.mod="gauss", start=namedlist, fit.marge=TRUE, loc.form=loc.form, scale.form=scale.form,shape.form=shape.form)

    # namedlist = list(cov11 = 1.0, cov12 = 1.2, cov22 = 2.2)
    # res = fitmaxstab(data=data, coord=coord, cov.mod="gauss", start=namedlist)

    # for (name in names(res)){
    #     print(name)
    #     print(res[name])
    # }
    print(res['fitted.values'])
    # print(res['convergence'])

}
