library(SpatialExtremes)

## Not run:
##Define the coordinate of each location
n.site <- 30
locations <- matrix(runif(2*n.site, 0, 10), ncol = 2)
colnames(locations) <- c("lon", "lat")

##  Fitting a penalized smoothing splines for the margins with the
##     Smith's model
data <- rmaxstab(100, locations, cov.mod = "gauss", cov11 = 100, cov12 =
                 25, cov22 = 220)

##     And transform it to ordinary GEV margins with a non-linear
##     function
fun <- function(x)
  2 * sin(pi * x / 4) + 10
fun2 <- function(x)
  (fun(x) - 7 ) / 15

param.loc <- fun(locations[,2])
param.scale <- fun(locations[,2])
param.shape <- fun2(locations[,1])

##Transformation from unit Frechet to common GEV margins
for (i in 1:n.site)
  data[,i] <- frech2gev(data[,i], param.loc[i], param.scale[i],
param.shape[i])

##Defining the knots, penalty, degree for the splines
n.knots_x = 2
n.knots_y = 2
knots = quantile(locations[,1], prob=1:n.knots_x/(n.knots_x+1))
knots2 = quantile(locations[,2], prob=1:n.knots_y/(n.knots_y+1))
knots_tot = cbind(knots,knots2)
print(knots)
print(knots2)
print(knots_tot)

##Be careful the choice of the penalty (i.e. the smoothing parameter)
##may strongly affect the result Here we use p-splines for each GEV
##parameter - so it's really CPU demanding but one can use 1 p-spline
##and 2 linear models.
##A simple linear model will be clearly faster...
loc.form <- y ~ rb(locations[,1], knots = knots, degree = 3, penalty = .5)
scale.form <- y ~ rb(locations[,2], knots = knots2, degree = 3, penalty = .5)
shape.form <- y ~ rb(locations, knots = knots_tot, degree = 3, penalty = .5)

fitted <- fitmaxstab(data, locations, "gauss", loc.form, scale.form, shape.form,
                     method = "BFGS")
fitted


## End(Not run)