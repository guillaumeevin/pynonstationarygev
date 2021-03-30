# Title     : TODO
# Objective : TODO
# Created by: erwan
# Created on: 30/03/2021

library(evgam)
library(SpatialExtremes)
# Sample from a GEV
set.seed(42)
N <- 50
loc = 0; scale = 1; shape <- 1
x_gev <- rgev(N, loc = loc, scale = scale, shape = shape)
# start_loc = 0; start_scale = 1; start_shape = 1
# N <- 50
# loc = 0; scale = 1; shape <- 0.1
# x_gev <- rgev(N, loc = loc, scale = scale, shape = shape)
coord <- matrix(ncol=1, nrow = N)
coord[,1]=seq(0,N-1,1)
colnames(coord) = c("T")
print(coord)
coord = data.frame(coord, stringsAsFactors = TRUE)
# res = fevd_fixed(x_gev, data=coord, method='MLE', verbose=TRUE, use.phi=FALSE)
# fmla = list(miaxi)
evgam()
# res = evgam(x_gev, data=coord, location.fun= ~T, scale.fun= ~T, method='MLE', type="GEV", verbose=FALSE, use.phi=FALSE)
# res = fevd_fixed(x_gev, data=coord, shape.fun= ~T, method='MLE', type="GEV", verbose=FALSE, use.phi=FALSE)
print(res)
