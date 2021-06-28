# Title     : TODO
# Objective : TODO
# Created by: erwan
# Created on: 30/03/2021
source('/home/erwan/Documents/projects/spatiotemporalextremes/extreme_fit/distribution/gev/evgam_fixed.R')
library(evgam)
library(mgcv)
library(SpatialExtremes)
data(COprcp)
set.seed(42)
N <- 101
loc = 0; scale = 1; shape <- 1
x_gev <- rgev(N, loc = loc, scale = scale, shape = shape)
years = runif(101)
indicator = c(rep(0, 51), rep(1, 50))
# years = seq(0, 100) / 100
df <- data.frame(x_gev, years, indicator)
colnames(df) <- c("prcp", "year", "indicator")
print(length(years))
# print(COprcp_gev)
print('before call')
fmla_gev2 <- list(prcp ~ s(year, k=4, m=1, bs="cr"), ~ 1, ~ 1)
# fmla_gev2 <- list(prcp ~ s(year, k=3, m=1, bs="cr"), ~ 1, ~ 1)
# fmla_gev2 <- list(prcp ~ s(elev, bs="bs", k=4, m=2), ~ 1, ~ 1)
m_gev2 <- evgam_fixed(fmla_gev2, data=df, family="gev")
# summary(m_gev2)
print('print results')
# print(m_gev2)
# print(m_gev2$coefficients)
location <- m_gev2$location
print(location)
# # print(location)
smooth <- m_gev2$location$smooth[[1]]
# # summary(location)
print(smooth)

# print(smooth[1])
# print(attr(smooth, "qrc"))
# m_gev2 <- evgam(fmla_gev2, data=COprcp_gev, family="gev")
# summary(m_gev2)
print(attributes(m_gev2))
print('good finish')
