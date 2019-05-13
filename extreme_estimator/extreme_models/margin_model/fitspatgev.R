# Title     : TODO
# Objective : TODO
# Created by: erwan
# Created on: 25/02/19
library(SpatialExtremes)



# fitspatgev_test with 2D data
fitspatgev_2D_test <- function (n.obs){
    n.site = 2
    coord <- matrix(rnorm(2*n.site, sd = sqrt(.2)), ncol = 2)
    colnames(coord) = c("E", "N")

    #  Generate the data
    data <- rmaxstab(n.obs, coord, "gauss", cov11 = 100, cov12 = 25, cov22 = 220)
    print(data)

    loc.form = loc ~ N
    scale.form = scale ~ 1
    shape.form = shape ~ 1

    namedlist = list(locCoeff1=0.0, locCoeff2=1.0, scaleCoeff1=0.1, shapeCoeff1=1.0)
    res = fitspatgev(data=data, covariables=coord, start=namedlist, loc.form=loc.form, scale.form=scale.form,shape.form=shape.form)
    print(res['fitted.values'])
}

# fitspatgev_test with 3D data
fitspatgev_3D_test_ok <- function (n.obs){
    n.site = 2
    covariables <- matrix(rnorm(2*n.site, sd = sqrt(.2)), ncol = 2)
    colnames(covariables) = c("E", "N")

    #  Generate the data
    data <- rmaxstab(n.obs, covariables, "gauss", cov11 = 100, cov12 = 25, cov22 = 220)
    print(data)

    loc.form = loc ~ N
    scale.form = scale ~ 1
    shape.form = shape ~ 1

    # Add the temporal covariates
    temp.form.loc = loc ~ T
    temp.form.scale = scale ~ T
    temp.form.shape = shape ~ T

    temp.cov = matrix(1:n.obs, ncol=1)
    colnames(temp.cov) = c("T")

    start = list(locCoeff1=0.0, locCoeff2=1.0, scaleCoeff1=0.1, shapeCoeff1=1.0,
                    tempCoeffLoc1=0.0, tempCoeffScale1=0.1, tempCoeffShape1=1.0)
    res = fitspatgev(data=data, covariables=covariables, start=start, loc.form=loc.form, scale.form=scale.form,shape.form=shape.form,
                temp.cov=temp.cov, temp.form.loc=temp.form.loc, temp.form.scale=temp.form.scale, temp.form.shape=temp.form.shape)


    print(res['fitted.values'])
}


# fitspatgev_test with 3D data
fitspatgev_3D_test <- function (n.obs){
    n.site = 2
    covariables <- matrix(rnorm(2*n.site, sd = sqrt(.2)), ncol = 2)
    colnames(covariables) = c("E", "N")

    #  Generate the data
    data <- rmaxstab(n.obs, covariables, "gauss", cov11 = 100, cov12 = 25, cov22 = 220)
    print(data)

    loc.form = loc ~ N
    scale.form = scale ~ N
    shape.form = shape ~ N

    # Add the temporal covariates
    temp.form.loc = loc ~ T
    temp.form.scale = scale ~ T
    # temp.form.shape = shape ~ T
    temp.form.shape = NULL

    temp.cov = matrix(1:n.obs, ncol=1)
    colnames(temp.cov) = c("T")

    # namedlist = list(locCoeff1=0.0, locCoeff2=1.0, scaleCoeff1=0.1, shapeCoeff1=1.0,
    #                 tempCoeffLoc1=0.0, tempCoeffLoc2=1.0, tempCoeffScale1=0.1, tempCoeffShape1=1.0)
    start = list(locCoeff1=0.0, locCoeff2=1.0, scaleCoeff1=0.1, shapeCoeff1=1.0,
                tempCoeffLoc1=0.0, tempCoeffLoc2=1.0, tempCoeffScale1=0.1, tempCoeffShape1=1.0)



    start = list(locCoeff1=0.0, locCoeff2=1.0, scaleCoeff1=0.1, shapeCoeff1=1.0, scaleCoeff2=0.1, shapeCoeff2=1.0,
                    tempCoeffLoc1=0.0, tempCoeffScale1=0.1, tempCoeffShape1=0.1)
        start = list(locCoeff1=0.0, locCoeff2=1.0, scaleCoeff1=0.1, shapeCoeff1=1.0, scaleCoeff2=0.1, shapeCoeff2=1.0,
                    tempCoeffLoc1=0.0, tempCoeffScale1=0.1)
    res = fitspatgev(data=data, covariables=covariables, start=start, loc.form=loc.form, scale.form=scale.form,shape.form=shape.form,
                temp.cov=temp.cov, temp.form.loc=temp.form.loc, temp.form.scale=temp.form.scale, temp.form.shape = temp.form.shape)

    n.site <- ncol(data)
    n.obs <- nrow(data)
    if (n.site != nrow(covariables))
        stop("'data' and 'covariates' doesn't match")
    use.temp.cov <- c(!is.null(temp.form.loc), !is.null(temp.form.scale),
        !is.null(temp.form.shape))
    if (any(use.temp.cov) && (n.obs != nrow(temp.cov)))
        stop("'data' and 'temp.cov' doesn't match")
    if (any(use.temp.cov) && is.null(temp.cov))
        stop("'temp.cov' must be supplied if at least one temporal formula is given")
    loc.form <- update(loc.form, y ~ .)
    scale.form <- update(scale.form, y ~ .)
    shape.form <- update(shape.form, y ~ .)
    print(use.temp.cov)
    if (use.temp.cov[1])
        temp.form.loc <- update(temp.form.loc, y ~ . + 0)
    if (use.temp.cov[2])
        temp.form.scale <- update(temp.form.scale, y ~ . + 0)
    if (use.temp.cov[3])
        temp.form.shape <- update(temp.form.shape, y ~ . + 0)
    loc.model <- modeldef(covariables, loc.form)
    scale.model <- modeldef(covariables, scale.form)
    shape.model <- modeldef(covariables, shape.form)
    loc.dsgn.mat <- loc.model$dsgn.mat
    scale.dsgn.mat <- scale.model$dsgn.mat
    shape.dsgn.mat <- shape.model$dsgn.mat
    loc.pen.mat <- loc.model$pen.mat
    scale.pen.mat <- scale.model$pen.mat
    shape.pen.mat <- shape.model$pen.mat
    loc.penalty <- loc.model$penalty.tot
    scale.penalty <- scale.model$penalty.tot
    shape.penalty <- shape.model$penalty.tot
    n.loccoeff <- ncol(loc.dsgn.mat)
    n.scalecoeff <- ncol(scale.dsgn.mat)
    n.shapecoeff <- ncol(shape.dsgn.mat)
    n.pparloc <- loc.model$n.ppar
    n.pparscale <- scale.model$n.ppar
    n.pparshape <- shape.model$n.ppar
    loc.names <- paste("locCoeff", 1:n.loccoeff, sep = "")
    scale.names <- paste("scaleCoeff", 1:n.scalecoeff, sep = "")
    shape.names <- paste("shapeCoeff", 1:n.shapecoeff, sep = "")
    if (use.temp.cov[1]) {
        temp.model.loc <- modeldef(temp.cov, temp.form.loc)
        temp.dsgn.mat.loc <- temp.model.loc$dsgn.mat
        temp.pen.mat.loc <- temp.model.loc$pen.mat
        temp.penalty.loc <- temp.model.loc$penalty.tot
        n.tempcoeff.loc <- ncol(temp.dsgn.mat.loc)
        n.ppartemp.loc <- temp.model.loc$n.ppar
        temp.names.loc <- paste("tempCoeffLoc", 1:n.tempcoeff.loc,
            sep = "")
    }
    else {
        temp.model.loc <- temp.dsgn.mat.loc <- temp.pen.mat.loc <- temp.names.loc <- NULL
        n.tempcoeff.loc <- n.ppartemp.loc <- temp.penalty.loc <- 0
    }
    if (use.temp.cov[2]) {
        temp.model.scale <- modeldef(temp.cov, temp.form.scale)
        temp.dsgn.mat.scale <- temp.model.scale$dsgn.mat
        temp.pen.mat.scale <- temp.model.scale$pen.mat
        temp.penalty.scale <- temp.model.scale$penalty.tot
        n.tempcoeff.scale <- ncol(temp.dsgn.mat.scale)
        n.ppartemp.scale <- temp.model.scale$n.ppar
        temp.names.scale <- paste("tempCoeffScale", 1:n.tempcoeff.scale,
            sep = "")
    }
    else {
        temp.model.scale <- temp.dsgn.mat.scale <- temp.pen.mat.scale <- temp.names.scale <- NULL
        n.tempcoeff.scale <- n.ppartemp.scale <- temp.penalty.scale <- 0
    }
    if (use.temp.cov[3]) {
        temp.model.shape <- modeldef(temp.cov, temp.form.shape)
        temp.dsgn.mat.shape <- temp.model.shape$dsgn.mat
        temp.pen.mat.shape <- temp.model.shape$pen.mat
        temp.penalty.shape <- temp.model.shape$penalty.tot
        n.tempcoeff.shape <- ncol(temp.dsgn.mat.shape)
        n.ppartemp.shape <- temp.model.shape$n.ppar
        temp.names.shape <- paste("tempCoeffShape", 1:n.tempcoeff.shape,
            sep = "")
    }
    else {
        temp.model.shape <- temp.dsgn.mat.shape <- temp.pen.mat.shape <- temp.names.shape <- NULL
        n.tempcoeff.shape <- n.ppartemp.shape <- temp.penalty.shape <- 0
    }
    param <- c(loc.names, scale.names, shape.names, temp.names.loc,
        temp.names.scale, temp.names.shape)
    print(param)
}



# Boolean for python call
call_main = !exists("python_wrapping")
if (call_main) {
    set.seed(42)
    n.obs = 10
    # fitspatgev_2D_test(n.obs)
    fitspatgev_3D_test(n.obs)

    # namedlist = list(cov11 = 1.0, cov12 = 1.2, cov22 = 2.2)
    # res = fitmaxstab(data=data, coord=coord, cov.mod="gauss", start=namedlist)

    # for (name in names(res)){
    #     print(name)
    #     print(res[name])
    # }

    # print(res['convergence'])

}


