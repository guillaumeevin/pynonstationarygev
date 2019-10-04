# Title     : TODO
# Objective : TODO
# Created by: erwan
# Created on: 04/10/2019
library(extRemes)
library(stats4)
library(SpatialExtremes)

set.seed(42)
N <- 1000
loc = 0; scale = 1; shape <- 0.1
x_gev <- rgev(N, loc = loc, scale = scale, shape = shape)
start_loc = 0; start_scale = 1; start_shape = 1
# res = fevd(x_gev, method='GMLE')

fevdPriorMy <- function (theta, q, p, log = FALSE){
    x = theta["shape"] + 0.5

    print(theta)
    print(theta["location"])
    print(dunif(theta["location"]))
    print(theta[0])
    dfun <- function(th) dbeta(th[1], shape1 = th[2], shape2 = th[3],
        log = log)
    th <- cbind(theta, q, p)
    res <- apply(th, 1, dfun)
    return(prod(res))
}

fevdPriorMyMy <- function (theta, q, p, log = FALSE){
    print(theta)
    print(q)
    print(p)
    x = theta[length(theta)]
    # + 0.5 enables to shift the Beta law in the interval [-0.5, 0.5]
    res = dbeta(x + 0.5, q, p, log = TRUE)
    return(res)
}


print(pbeta(1.0, 1, 1))
print(pbeta(0.5, 1, 1))
print(fevdPriorMy(2.0, 0.0, 0.0))

res = fevd(x_gev, method='Bayesian', priorFun="fevdPriorMyMy", priorParams=list(q=c(6), p=c(9)), iter=5000)
print(res)
# res = fevd(x_gev, method='Bayesian')
# print(res)

priorFun="shapePriorBeta"
shapePriorBeta
#
#
# print(shapePriorBeta(0.0, 6, 9))
# priorParams=list(q=c(1, 1, 6), p=c(1, 1, 9))
# p.i <- do.call(priorFun, c(list(1.0), priorParams))
# print(p.i)

# priorFun <- "shapePriorBeta"
# priorParams <- list(q = 6, p = 9)
# priorFun <- "fevdPriorDefault"
# priorParams <- list(q = 6, p = 9)
# e = do.call(priorFun, c(list(0.0), priorParams))
# print(e)
#
# print(res$method)
# print(res$priorFun)
# print(res$priorParams)
# m = res$results
# print(m[2,1])
# print(class(res$chain.info))
# print(res$chain.info[[1]])
# # summary(res)
# print(attributes(res))
# print('here')
# print(attr(res, 'chain.info'))
# print(attr(res, "method"))
# print(attr(res, "x"))
# print(attr(res, "priorParams"))

# print(res.method)


# p.i <- do.call(shapePriorBeta, c(list(theta =  c(-0.12572432087762, -0.0567634605386987, 0.133782230298093)), priorParams=list(q = 6, p = 9)))
# print(p.i)
# a = fevd(x_gev, method='Bayesian', priorFun="shapePriorBeta", priorParams=list(q = 6, p = 9))

# priorParams=list(v=c(0.1, 10, 0.1)),
#     initial=list(location=0, scale=0.1, shape=-0.5)),

# print(a)
#
# # S3 method for fevd.bayesian
# summary(a, FUN = "mean", burn.in = 499)

# print(a.results)

# Bayesian method is using a normal distribution functions for the shape parameter
# GMLE distribution is using a Beta distribution for the shape parameter
