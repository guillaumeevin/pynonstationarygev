library(SpatialExtremes)
# Title     : TODO
# Objective : TODO
# Created by: erwan
# Created on: 27/11/18
# define the coordinate of each location
set.seed(42)
n.site <- 30
locations <- matrix(runif(2*n.site, 0, 10), ncol = 2)
colnames(locations) <- c("lon", "lat")
print(locations)
##Simulate a max-stable process - with unit Frechet margins
data <- rmaxstab(40, locations, cov.mod = "whitmat", nugget = 0, range = 3,
smooth = 0.5)
##Now define the spatial model for the GEV parameters
param.loc <- -10 + 2 * locations[,2]
param.scale <- 5 + 2 * locations[,1] + locations[,2]^2
param.shape <- rep(0.2, n.site)
##Transform the unit Frechet margins to GEV
for (i in 1:n.site)
data[,i] <- frech2gev(data[,i], param.loc[i], param.scale[i], param.shape[i])
##Define a model for the GEV margins to be fitted
##shape ~ 1 stands for the GEV shape parameter is constant
##over the region
loc.form <- loc ~ lat
scale.form <- scale ~ lon + I(lat^2)
shape.form <- shape ~ 1
##Fit a max-stable process using the Schlather
res = fitmaxstab(data, locations, "whitmat", loc.form, scale.form, shape.form)
print(res['fitted.values'])
## Model without any spatial structure for the GEV parameters
## Be careful this could be *REALLY* time consuming
# res = fitmaxstab(data, locations, "whitmat", fit.margin=TRUE)
# print(res['fitted.values'])

