library(SpatialExtremes)



# rmaxstab with 2D data
rmaxstab2D <- function (n.obs){
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

    loc.form = loc ~ N
    scale.form = scale ~ 1
    shape.form = shape ~ 1

    namedlist = list(cov11 = 1.0, cov12 = 1.2, cov22 = 2.2, locCoeff1=1.0, locCoeff2=1.0, scaleCoeff1=2.0, shapeCoeff1=0.1)
    res = fitmaxstab(data=data, coord=coord, cov.mod="gauss", start=namedlist, fit.marge=TRUE, loc.form=loc.form, scale.form=scale.form,shape.form=shape.form)
    print(res['fitted.values'])
}

# rmaxstab with 3D data
rmaxstab3Dimprovedgauss <- function (n.obs){
    n.site = 2
    coord_sample <- matrix(rnorm(2*n.site, sd = sqrt(.2)), ncol = 2)
    colnames(coord_sample) = c("E", "N")

    #  Generate the data
    data <- rmaxstab(n.obs, coord_sample, "gauss", cov11 = 100, cov12 = 25, cov22 = 220)
    # data <- rmaxstab(n.obs, coord, "brown", range = 3, smooth = 0.5)
    # data <- rmaxstab(n.obs, coord, "whitmat", nugget = 0.0, range = 3, smooth = 0.5)
    #  Fit back the data
    # print(data)n
    # res = fitmaxstab(data, coord, "gauss", fit.marge=FALSE, )
    # res = fitmaxstab(data, coord, "brown")
    # res = fitmaxstab(data, coord, "whitmat", start=)

    coord_fit <- matrix(rnorm(3*n.site, sd = sqrt(.2)), ncol = 3)
    colnames(coord_fit) = c("E", "N", "A")

    print(class(coord_fit))
    print(colnames(coord_fit))

    loc.form = loc ~ N
    scale.form = scale ~ 1
    shape.form = shape ~ 1

    namedlist = list(cov11 = 1.0, cov12 = 1.2, cov22 = 2.2, cov13 = 1.0, cov23 = 1.2, cov33 = 2.2, locCoeff1=1.0, locCoeff2=1.0, scaleCoeff1=2.0, shapeCoeff1=0.1)
    res = fitmaxstab(data=data, coord=coord_fit, cov.mod="gauss", start=namedlist, fit.marge=TRUE, loc.form=loc.form, scale.form=scale.form,shape.form=shape.form)
    print(res['fitted.values'])
}

rmaxstab3Dimprovedbrown <- function (n.obs){
    n.site = 2
    coord_sample <- matrix(rnorm(2*n.site, sd = sqrt(.2)), ncol = 2)
    colnames(coord_sample) = c("E", "N")

    #  Generate the data
    data <- rmaxstab(n.obs, coord_sample, "gauss", cov11 = 100, cov12 = 25, cov22 = 220)
    # data <- rmaxstab(n.obs, coord, "brown", range = 3, smooth = 0.5)
    # data <- rmaxstab(n.obs, coord, "whitmat", nugget = 0.0, range = 3, smooth = 0.5)
    #  Fit back the data
    # print(data)n
    # res = fitmaxstab(data, coord, "gauss", fit.marge=FALSE, )
    # res = fitmaxstab(data, coord, "brown")
    # res = fitmaxstab(data, coord, "whitmat", start=)

    coord_fit <- matrix(rnorm(3*n.site, sd = sqrt(.2)), ncol = 3)
    colnames(coord_fit) = c("E", "N", "A")

    print(class(coord_fit))
    print(colnames(coord_fit))

    loc.form = loc ~ N
    scale.form = scale ~ 1
    shape.form = shape ~ 1

    # namedlist = list(cov11 = 1.0, cov12 = 1.2, cov22 = 2.2, cov13 = 1.0, cov23 = 1.2, cov33 = 2.2, locCoeff1=1.0, locCoeff2=1.0, scaleCoeff1=2.0, shapeCoeff1=0.1)
    res = fitmaxstab(data=data, coord=coord_fit, cov.mod="brown", fit.marge=TRUE, loc.form=loc.form, scale.form=scale.form,shape.form=shape.form)
    print(res['fitted.values'])
}

rmaxstab4Dimprovedbrown <- function (n.obs){
    n.site = 2
    coord_sample <- matrix(rnorm(2*n.site, sd = sqrt(.2)), ncol = 2)
    colnames(coord_sample) = c("E", "N")

    #  Generate the data
    data <- rmaxstab(n.obs, coord_sample, "gauss", cov11 = 100, cov12 = 25, cov22 = 220)
    # data <- rmaxstab(n.obs, coord, "brown", range = 3, smooth = 0.5)
    # data <- rmaxstab(n.obs, coord, "whitmat", nugget = 0.0, range = 3, smooth = 0.5)
    #  Fit back the data
    # print(data)n
    # res = fitmaxstab(data, coord, "gauss", fit.marge=FALSE, )
    # res = fitmaxstab(data, coord, "brown")
    # res = fitmaxstab(data, coord, "whitmat", start=)

    coord_fit <- matrix(rnorm(4*n.site, sd = sqrt(.2)), ncol = 4)
    colnames(coord_fit) = c("E", "N", "A", "B")

    print(class(coord_fit))
    print(colnames(coord_fit))

    loc.form = loc ~ N + E + A + B
    scale.form = scale ~ 1
    shape.form = shape ~ 1

    # namedlist = list(cov11 = 1.0, cov12 = 1.2, cov22 = 2.2, cov13 = 1.0, cov23 = 1.2, cov33 = 2.2, locCoeff1=1.0, locCoeff2=1.0, scaleCoeff1=2.0, shapeCoeff1=0.1)
    res = fitmaxstab(data=data, coord=coord_fit, cov.mod="brown", fit.marge=TRUE, loc.form=loc.form, scale.form=scale.form,shape.form=shape.form)
    print(res['fitted.values'])
}


rmaxstab3Dimprovedtpowexp <- function (n.obs){
    n.site = 2
    coord_sample <- matrix(rnorm(2*n.site, sd = sqrt(.2)), ncol = 2)
    colnames(coord_sample) = c("E", "N")

    #  Generate the data
    data <- rmaxstab(n.obs, coord_sample, "gauss", cov11 = 100, cov12 = 25, cov22 = 220)
    # data <- rmaxstab(n.obs, coord, "brown", range = 3, smooth = 0.5)
    # data <- rmaxstab(n.obs, coord, "whitmat", nugget = 0.0, range = 3, smooth = 0.5)
    #  Fit back the data
    # print(data)n
    # res = fitmaxstab(data, coord, "gauss", fit.marge=FALSE, )
    # res = fitmaxstab(data, coord, "brown")
    # res = fitmaxstab(data, coord, "whitmat", start=)

    coord_fit <- matrix(rnorm(3*n.site, sd = sqrt(.2)), ncol = 3)
    colnames(coord_fit) = c("E", "N", "A")

    print(class(coord_fit))
    print(colnames(coord_fit))

    loc.form = loc ~ N
    scale.form = scale ~ 1
    shape.form = shape ~ 1

    # namedlist = list(cov11 = 1.0, cov12 = 1.2, cov22 = 2.2, cov13 = 1.0, cov23 = 1.2, cov33 = 2.2, locCoeff1=1.0, locCoeff2=1.0, scaleCoeff1=2.0, shapeCoeff1=0.1)
    res = fitmaxstab(data=data, coord=coord_fit, cov.mod="tpowexp", fit.marge=TRUE, loc.form=loc.form, scale.form=scale.form,shape.form=shape.form)
    print(res['fitted.values'])
}



# rmaxstab with 3D data
# rmaxstab3D <- function (n.obs){
#     # todo: problem this function is currently not available in dimensions 3
#     n.site = 2
#     dimension = 3
#     ar = rnorm(3*n.obs*n.site, sd = sqrt(.2))
#     print(ar)
#     coord <- array(ar, dim = c(4,3))
#     print(coord)
#     colnames(coord) = c("E", "N", "T")
#     print(colnames(coord))
#     data <- coord
#
#     #  Generate the data
#     # data <- rmaxstab(n.obs, coord, "gauss", cov11 = 100, cov12 = 25, cov22 = 220)
#     # data <- rmaxstab(n.obs, coord, "brown", range = 3, smooth = 0.5)
#     # data <- rmaxstab(n.obs, coord, "whitmat", nugget = 0.0, range = 3, smooth = 0.5)
#     #  Fit back the data
#     # print(data)n
#     # res = fitmaxstab(data, coord, "gauss", fit.marge=FALSE, )
#     # res = fitmaxstab(data, coord, "brown")
#     # res = fitmaxstab(data, coord, "whitmat", start=)
#     print(class(coord))
#     print(colnames(coord))
#
#     loc.form = loc ~ N
#     scale.form = scale ~ 1
#     shape.form = shape ~ 1
#
#     temp_loc.form = loc ~ T
#     temp_scale.form = scale ~ 1
#     temp_shape.form = shape ~ 1
#
#     namedlist = list(cov11 = 1.0, cov12 = 1.2, cov22 = 2.2, locCoeff1=1.0, locCoeff2=1.0, scaleCoeff1=1.0, shapeCoeff1=1.0)
#     res = fitmaxstab(data=data, coord=coord, cov.mod="gauss", start=namedlist, fit.marge=TRUE,
#     loc.form=loc.form, scale.form=scale.form,shape.form=shape.form,
#     temp.loc.form = temp_loc.form, temp.scale.form = temp_scale.form, temp.shape.form = temp_shape.form)
#     print(res['fitted.values'])
# }


# rmaxstab with 1D data
rmaxstab1D <- function (n.obs){

    # In one dimensional, we duplicate the coordinate
    vec = rnorm(3, sd = sqrt(.2))
    coord = cbind(vec, vec)
    var = 1.0
    data <- rmaxstab(n.obs, coord, "gauss", cov11 = var, cov12 = 0, cov22 = var)

    print(class(coord))
    print(colnames(coord))

    loc.form = loc ~ 1
    scale.form = scale ~ 1
    shape.form = shape ~ 1

    # GAUSS
    namedlist = list(cov=1.0, locCoeff1=1.0, scaleCoeff1=1.0, shapeCoeff1=1.0)
    res = fitmaxstab(data=data, coord=coord, cov.mod="gauss", start=namedlist, fit.marge=TRUE, loc.form=loc.form, scale.form=scale.form,shape.form=shape.form, iso=TRUE, control=list(maxit=1000))

    # ‘eval.max’
    # ‘iter.max’

    # BROWN
    # namedlist = list(range = 3, smooth = 0.5, locCoeff1=1.0, scaleCoeff1=1.0, shapeCoeff1=1.0)
    # res = fitmaxstab(data=data, coord=coord, cov.mod="brown", start=namedlist, fit.marge=TRUE, loc.form=loc.form, scale.form=scale.form,shape.form=shape.form, iso=TRUE)


    print(res['fitted.values'])
}

# Boolean for python call
call_main = !exists("python_wrapping")
if (call_main) {
    set.seed(42)
    n.obs = 50
    # rmaxstab3Dimprovedgauss(n.obs)
    rmaxstab4Dimprovedbrown(n.obs)
    # rmaxstab3Dimprovedbrown(n.obs)
    # rmaxstab3Dimprovedtpowexp(n.obs)
    # rmaxstab2D(n.obs)
    # rmaxstab3D(n.obs)
    # rmaxstab1D(n.obs)

    # namedlist = list(cov11 = 1.0, cov12 = 1.2, cov22 = 2.2)
    # res = fitmaxstab(data=data, coord=coord, cov.mod="gauss", start=namedlist)

    # for (name in names(res)){
    #     print(name)
    #     print(res[name])
    # }

    # print(res['convergence'])

}

