# Title     : TODO
# Objective : TODO
# Created by: erwan
# Created on: 04/10/2019
library(extRemes)
library(data.table)
library(stats4)
library(SpatialExtremes)
source('fevd_fixed.R')
source('ci_fevd_fixed.R')
# Sample from a GEV
set.seed(42)
N <- 50
loc = 0; scale = 1; shape <- 0.1
x_gev <- rgev(N, loc = loc, scale = scale, shape = shape)
coord <- matrix(ncol=1, nrow = N)
coord[,1]=seq(0,N-1,1)
colnames(coord) = c("T")
coord = data.frame(coord, stringsAsFactors = TRUE)
# res = fevd_fixed(x_gev, data=coord, method='MLE', verbose=TRUE, use.phi=FALSE)
res = fevd_fixed(x_gev, data=coord, location.fun= ~T, method='MLE', type="Gumbel", verbose=FALSE, use.phi=FALSE)
print(res)

# Some display for the results
# m = res$results
# print(class(res$chain.info))
# print(dim(m))
# print(m)
print(res$results$par)
# print(res$par)
# print(m[1])


# Confidence interval staionary
# method = "proflik"
# res_ci = ci.fevd.mle(res, alpha = 0.05, type = c("return.level"),
#     return.period = 50, method = method, xrange = c(-200,200), nint = 10, R=502, verbose = TRUE,
#     tscale = FALSE, return.samples = FALSE)
# print(res_ci)

# Bug to solve for the non stationary - the returned parameter do not match with the return level
v = make.qcov(res, vals = list(mu1 = c(0.0)))
print(res$results$num.pars$location)
print(res$results$num.pars$scale)
print(res$results$num.pars$shape)
res$results$num.pars$shape = 0
print(res$results$num.pars$shape)

# r = return.level(res, return.period = 50, qcov = v, type='Gumbel')
# print(r)
# param = findpars(res, qcov = v)
# print(param)
# print(res$results$num.pars$shape)
res_ci = ci.fevd.mle_fixed(res, alpha = 0.05, type = c("return.level"),
    return.period = 50, method = "normal", xrange = NULL, nint = 20, verbose = FALSE,
    tscale = FALSE, return.samples = FALSE, qcov=v)
print(res_ci)







