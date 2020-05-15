# Title     : TODO
# Objective : TODO
# Created by: erwan
# Created on: 15/05/2020
# library(rmutil)
library(evd)
library(ismev)
source("gnfit_fixed.R")
data = rgumbel(10)
res = gnfit_fixed(data, "gum")
print(res)
# r = rnorm(0)
# disp(r)
# rGumbel(20)
