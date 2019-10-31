# Title     : TODO
# Objective : TODO
# Created by: erwan
# Created on: 04/10/2019
library(extRemes)
library(data.table)
library(stats4)
library(SpatialExtremes)
source('fevd_fixed.R')
# Sample from a GEV
set.seed(42)
N <- 50
loc = 0; scale = 1; shape <- 1
x_gev <- rgev(N, loc = loc, scale = scale, shape = shape)
coord <- matrix(ncol=1, nrow = N)
coord[,1]=seq(0,N-1,1)
colnames(coord) = c("T")
coord = data.frame(coord, stringsAsFactors = TRUE)
res = fevd_fixed(x_gev, data=coord, method='MLE', verbose=TRUE, use.phi=FALSE)
# res = fevd_fixed(x_gev, data=coord, location.fun= ~T, method='MLE', verbose=TRUE, use.phi=FALSE, time.units = "years", units = "years")
# print(res)

# Some display for the results
# m = res$results
# print(class(res$chain.info))
# print(dim(m))
# print(m)
print(res$results$par)
# print(res$par)
# print(m[1])


# Confidence interval staionary
method = "normal"
res_ci = ci(res, alpha = 0.05, type = c("return.level", "parameter"),
    return.period = 50, method = method, xrange = NULL, nint = 20, verbose = FALSE,
    tscale = FALSE)
print(res_ci)

# Bug to solve for the non stationary - the returned parameter do not match with the return level
# ci.fevd.mle()
# Confidence interval non staionary
# v = make.qcov(res, vals = list(mu1 = c(0.0)))
# r = return.level(res, return.period = 50, qcov = v)
# print(r)
# param = findpars(res, qcov = v)
# print(param)
# res_ci = ci(res, alpha = 0.05, type = c("return.level", "parameter"),
#     return.period = 50, method = "normal", xrange = NULL, nint = 20, verbose = FALSE,
#     tscale = FALSE, return.samples = FALSE, qcov=v)
# print(res_ci)
#
#





