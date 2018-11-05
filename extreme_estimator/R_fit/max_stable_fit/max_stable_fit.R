library(SpatialExtremes)


# Boolean for python call
call_main = !exists("python_wrapping")

if (call_main) {
    set.seed(42)
    n.obs = 50
    n.site = 2
    coord <- matrix(rnorm(2*n.site, sd = sqrt(.2)), ncol = 2)

    print(coord)
    #  Generate the data
    # data <- rmaxstab(n.obs, coord, "gauss", cov11 = 100, cov12 = 25, cov22 = 220)
    # data <- rmaxstab(n.obs, coord, "brown", range = 3, smooth = 0.5)
    # data <- rmaxstab(n.obs, coord, "whitmat", nugget = 0.0, range = 3, smooth = 0.5)
    data <- rmaxstab(n.obs, coord, "cauchy", )
    #  Fit back the data
    print(data)
    # res = fitmaxstab(data, coord, "gauss", fit.marge=FALSE, )
    # res = fitmaxstab(data, coord, "brown")
    res = fitmaxstab(data, coord, "cauchy")
    # res = fitmaxstab(data, coord, "gauss", start=list(0,0,0))
    print(res)
    print(class(res))
    print(names(res))
    print(res['fitted.values'])
    print(res['convergence'])

}
