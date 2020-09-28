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
source('summary_fevd_fixed.R')
# Sample from a GEV
set.seed(42)
N <- 100
loc = 0; scale = 1; shape <- 0.1
x_gev <- rgev(N, loc = loc, scale = scale, shape = shape)
# start_loc = 0; start_scale = 1; start_shape = 1
# N <- 50
# loc = 0; scale = 1; shape <- 0.1
# x_gev <- rgev(N, loc = loc, scale = scale, shape = shape)
print(N)
coord <- matrix(ncol=2, nrow = N)
coord[,1]=seq(0,N-1,1)
coord[,2]=seq(0,N-1,1)

colnames(coord) = c("X", "T")
coord = data.frame(coord, stringsAsFactors = TRUE)
# res = fevd_fixed(x_gev, data=coord, method='MLE', verbose=TRUE, use.phi=FALSE)
# res = fevd_fixed(x_gev, data=coord, location.fun= ~T, scale.fun= ~T, method='MLE', type="GEV", verbose=FALSE, use.phi=FALSE)
# res = fevd_fixed(x_gev, data=coord, location.fun= ~sin(X) + cos(T), method='MLE', type="GEV", verbose=FALSE, use.phi=FALSE)
# res = fevd_fixed(x_gev, data=coord, location.fun= ~poly(X * T, 1, raw = TRUE),  method='MLE', type="Gumbel", verbose=FALSE, use.phi=FALSE)
res = fevd_fixed(x_gev, data=coord, location.fun= ~poly(X, 1, raw = TRUE) + poly(T, 1, raw = TRUE) , method='MLE', type="Gumbel", verbose=FALSE, use.phi=FALSE)
# print(res)
print(summary.fevd.mle_fixed(res))
# print(summary(res)$AIC)

# Some display for the results
# m = res$results
# print(class(res$chain.info))
# print(dim(m))
# print(m)
# print(res$results$par)
# print(res$par)
# print(m[1])


# v = make.qcov(res, vals = list(mu0 = c(0.0), mu1 = c(0.0), mu2 = c(0.0), sigma = c(0.0)))
v = make.qcov(res, vals = list(mu1 = c(0.0), mu2 = c(0.0)))

res_ci = ci.fevd.mle_fixed(res, alpha = 0.05, type = c("return.level"),
    return.period = 50, method = "normal", xrange = NULL, nint = 20, verbose = FALSE,
    tscale = FALSE, return.samples = FALSE, qcov=v)
print(res_ci)





# Confidence interval staionary
# method = "proflik"
# res_ci = ci.fevd.mle(res, alpha = 0.05, type = c("return.level"),
#     return.period = 50, method = method, xrange = c(-200,200), nint = 10, R=502, verbose = TRUE,
#     tscale = FALSE, return.samples = FALSE)
# print(res_ci)

# Bug to solve for the non stationary - the returned parameter do not match with the return level








