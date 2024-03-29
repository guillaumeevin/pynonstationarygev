# Title     : TODO
# Objective : TODO
# Created by: erwan
# Created on: 06/05/2019

library(stats4)
library(ismev)
library(SpatialExtremes)

# Boolean for python call
call_main = !exists("python_wrapping")

if (call_main) {
    set.seed(42)
    N <- 50
    loc = 0; scale = 1; shape <- 1
    x_gev <- rgev(N, loc = loc, scale = scale, shape = shape)
    start_loc = 0; start_scale = 1; start_shape = 1
    # mul = NULL
    # y = NULL
    mul = 1
    y =matrix(ncol=1,nrow=50)
    y[,1]=seq(1,50,1)
}




gev_fit_copy <- function (xdat, ydat = NULL, mul = NULL, sigl = NULL, shl = NULL,
    mulink = identity, siglink = identity, shlink = identity,
    muinit = NULL, siginit = NULL, shinit = NULL, show = TRUE,
    method = "Nelder-Mead", maxit = 10000, ...)
{
    z <- list()
    npmu <- length(mul) + 1
    npsc <- length(sigl) + 1
    npsh <- length(shl) + 1
    z$trans <- FALSE
    in2 <- sqrt(6 * var(xdat))/pi
    in1 <- mean(xdat) - 0.57722 * in2
    if (is.null(mul)) {
        mumat <- as.matrix(rep(1, length(xdat)))
        if (is.null(muinit))
            muinit <- in1
    }
    else {
        z$trans <- TRUE
        mumat <- cbind(rep(1, length(xdat)), ydat[, mul])
        if (is.null(muinit))
            muinit <- c(in1, rep(0, length(mul)))
    }
    if (is.null(sigl)) {
        sigmat <- as.matrix(rep(1, length(xdat)))
        if (is.null(siginit))
            siginit <- in2
    }
    else {
        z$trans <- TRUE
        sigmat <- cbind(rep(1, length(xdat)), ydat[, sigl])
        if (is.null(siginit))
            siginit <- c(in2, rep(0, length(sigl)))
    }
    if (is.null(shl)) {
        shmat <- as.matrix(rep(1, length(xdat)))
        if (is.null(shinit))
            shinit <- 0.1
    }
    else {
        z$trans <- TRUE
        shmat <- cbind(rep(1, length(xdat)), ydat[, shl])
        if (is.null(shinit))
            shinit <- c(0.1, rep(0, length(shl)))
    }
    z$model <- list(mul, sigl, shl)
    z$link <- deparse(substitute(c(mulink, siglink, shlink)))
    init <- c(muinit, siginit, shinit)
    gev.lik <- function(a) {
        mu <- mulink(mumat %*% (a[1:npmu]))
        sc <- siglink(sigmat %*% (a[seq(npmu + 1, length = npsc)]))
        xi <- shlink(shmat %*% (a[seq(npmu + npsc + 1, length = npsh)]))
        print(class(xdat))
        print(class(mu))
        y <- (xdat - mu)/sc
        y <- 1 + xi * y
        if (any(y <= 0) || any(sc <= 0))
            return(10^6)
        sum(log(sc)) + sum(y^(-1/xi)) + sum(log(y) * (1/xi +
            1))
    }
    x <- optim(init, gev.lik, hessian = TRUE, method = method,
        control = list(maxit = maxit, ...))
    z$conv <- x$convergence
    mu <- mulink(mumat %*% (x$par[1:npmu]))
    sc <- siglink(sigmat %*% (x$par[seq(npmu + 1, length = npsc)]))
    xi <- shlink(shmat %*% (x$par[seq(npmu + npsc + 1, length = npsh)]))
    z$nllh <- x$value
    z$data <- xdat
    if (z$trans) {
        z$data <- -log(as.vector((1 + (xi * (xdat - mu))/sc)^(-1/xi)))
    }
    z$mle <- x$par
    z$cov <- solve(x$hessian)
    z$se <- sqrt(diag(z$cov))
    z$vals <- cbind(mu, sc, xi)
    if (show) {
        if (z$trans)
            print(z[c(2, 3, 4)])
        else print(z[4])
        if (!z$conv)
            print(z[c(5, 7, 9)])
    }
    class(z) <- "gev.fit"
    invisible(z)
}

gev_fit <- function(data, y = NULL, trend=NULL){
    params = gev_fit_copy(data, y, mul=trend)
    return(params)
}

main <- function(){
    param = gev_fit(x_gev, y, mul)
}

if (call_main) {
    main()
}
