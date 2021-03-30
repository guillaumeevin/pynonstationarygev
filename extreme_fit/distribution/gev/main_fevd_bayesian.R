# Title     : TODO
# Objective : TODO
# Created by: erwan
# Created on: 04/10/2019
library(extRemes)
library(data.table)
library(stats4)
library(quantreg)
library(SpatialExtremes)
source('fevd_fixed.R')
# Sample from a GEV
set.seed(42)
N <- 50
loc = 0; scale = 1; shape <- 1
x_gev <- rgev(N, loc = loc, scale = scale, shape = shape)
start_loc = 0; start_scale = 1; start_shape = 1

# fevdPriorMy <- function (theta, q, p, log = FALSE){
#     x = theta["shape"] + 0.5
#
#     print(theta)
#     print(theta["location"])
#     print(dunif(theta["location"]))
#     print(theta[0])
#     dfun <- function(th) dbeta(th[1], shape1 = th[2], shape2 = th[3],
#         log = log)
#     th <- cbind(theta, q, p)
#     res <- apply(th, 1, dfun)
#     return(prod(res))
# }



# print(pbeta(1.0, 1, 1))
# print(pbeta(0.5, 1, 1))
# print(fevdPriorCustom(2.0, 0.0, 0.0))



# res = fevd(x_gev, method='Bayesian', priorFun="fevdPriorMyMy", priorParams=list(q=c(6), p=c(9)), iter=5000, verbose=TRUE, use.phi=FALSE)
# res = fevd(x_gev, method='GMLE', iter=5000, verbose=TRUE, use.phi=FALSE)

# Without covariate
# res = fevd_fixed(x_gev, method='Bayesian', priorFun="fevdPriorCustom", priorParams=list(q=c(6), p=c(9)), iter=5000, verbose=TRUE, use.phi=FALSE)

# Add covariate
coord <- matrix(ncol=1, nrow = N)
coord[,1]=seq(0,N-1,1)
print(coord)
colnames(coord) = c("T")
coord = data.frame(coord, stringsAsFactors = TRUE)
res = fevd_fixed(x_gev, data=coord, location.fun= ~T, method='Bayesian', priorFun="fevdPriorCustom",
priorParams=list(q=c(6), p=c(9)), iter=5000, verbose=TRUE, use.phi=FALSE,
initial=list())
# res = fevd_fixed(x_gev, data=coord, method='Bayesian', priorFun="fevdPriorCustom", priorParams=list(q=c(6), p=c(9)), iter=5000, verbose=TRUE, use.phi=FALSE)
print(res)

v = make.qcov(res, vals = list(mu1 = c(0.0)))
# # res_find = findpars(res, FUN = "mean" , tscale = FALSE, qcov = v,
# #     burn.in=4998)
# # print(res_find)
# #
res_ci = ci(res, alpha = 0.05, type = c("return.level"),
    burn.in=499,
    return.period = 50, FUN = "mean", tscale = FALSE,  qcov=v)
print(res_ci)
print(res_ci[1, ])


# Some display for the results
# print(res)
# print('here')
# print(res$constant.loc)
# print('here2')
# print(res$method)
# print(res$priorFun)
# print(res$priorParams)
# m = res$results
# print(class(res$chain.info))
# print(dim(m))
# print(m[1,])
# print(m[1,1])
# print(res$chain.info[1,])



# Confidence interval
# res_ci = ci(res, alpha = 0.05, type = c("return.level"),
#     return.period = 50, FUN = "mean", tscale = FALSE)
# print(res_ci)
# print(attributes(res_ci))
# print(dimnames(res_ci))
# print(res_ci[1])


# covariates = c(1.0, 100.0, 1000.0)
# v = make.qcov(res, vals = list(mu1 = covariates, sigma1 = covariates))
# v = make.qcov(res, vals = list(mu1 = c(0.0)))
# # res_find = findpars(res, FUN = "mean" , tscale = FALSE, qcov = v,
# #     burn.in=4998)
# # print(res_find)
# #
# res_ci = ci(res, alpha = 0.05, type = c("return.level"),
#     burn.in=4998,
#     return.period = 50, FUN = "mean", tscale = FALSE,  qcov=v)
# print(res_ci)
# print(res_ci[1, ])
#
# r = qgev(0.98, 0.1427229 , 1.120554, -0.1008511)
# print(r)
# print(attributes(res_ci))
# print(dimnames(res_ci))
# print(res_ci[1])
# print(attr(res_ci), '')





