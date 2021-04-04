# Title     : TODO
# Objective : TODO
# Created by: erwan
# Created on: 30/03/2021
source('evgam_fixed.R')
library(evgam)
data(COprcp)
COprcp$year <- format(COprcp$date, "%Y")
COprcp_gev <- aggregate(prcp ~ year + meta_row, COprcp, max)
COprcp_gev <- cbind(COprcp_gev, COprcp_meta[COprcp_gev$meta_row,])
# print(COprcp_gev)
print('before call')
# fmla_gev2 <- list(prcp ~ s(elev, k=3, m=1), ~ 1, ~ 1)
fmla_gev2 <- list(prcp ~ s(elev, bs="bs", k=4, m=2), ~ 1, ~ 1)
m_gev2 <- evgam_fixed(fmla_gev2, data=COprcp_gev, family="gev")
print('print results')
print(m_gev2)
# print(m_gev2$coefficients)
location <- m_gev2$location
print(location)
# print(location$coefficients)
# smooth <- m_gev2$location$smooth
# print(smooth)
# print(attr(smooth, "qrc"))
# print(m_gev2.knots)
# m_gev2 <- evgam(fmla_gev2, data=COprcp_gev, family="gev")
# summary(m_gev2)
# print(attributes(m_gev2))
print('good finish')
