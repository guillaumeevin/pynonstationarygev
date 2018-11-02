library(SpatialExtremes)
##Define the spatial_coordinates of each location
n.site <- 30
locations <- matrix(runif(2*n.site, 0, 10), ncol = 2)
colnames(locations) <- c("lon", "lat")

##Simulate a max-stable process - with unit Frechet margins
data <- rmaxstab(40, locations, cov.mod = "whitmat", nugget = 0, range = 3,
smooth = 0.5)

##Now define the spatial model for the GEV parameters
param.loc <- -10 + 2 * locations[,2]
param.scale <- 5 + 2 * locations[,1] + locations[,2]^2
param.shape <- rep(0.2, n.site)

##Transform the unit Frechet margins to GEV
for (i in 1:n.site)
  data[,i] <- frech2gev(data[,i], param.loc[i], param.scale[i],
param.shape[i])

##Define a model for the GEV margins to be fitted
##shape ~ 1 stands for the GEV shape parameter is constant
##over the region
loc.form <- loc ~ lat
scale.form <- scale ~ lon + I(lat^2)
shape.form <- shape ~ 1

##Fit a max-stable process using the Schlather's model
fitmaxstab(data, locations, "whitmat", loc.form, scale.form,
           shape.form)

## Model without any spatial structure for the GEV parameters
## Be careful this could be *REALLY* time consuming
fitmaxstab(data, locations, "whitmat")

##  Fixing the smooth parameter of the Whittle-Matern family
##  to 0.5 - e.g. considering exponential family. We suppose the data
##  are unit Frechet here.
fitmaxstab(data, locations, "whitmat", smooth = 0.5, fit.marge = FALSE)

##  Fitting a penalized smoothing splines for the margins with the
##     Smith's model
data <- rmaxstab(40, locations, cov.mod = "gauss", cov11 = 100, cov12 =
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
n.knots <- 5
knots <- quantile(locations[,2], prob = 1:n.knots/(n.knots+1))
knots2 <- quantile(locations[,1], prob = 1:n.knots/(n.knots+1))

##Be careful the choice of the penalty (i.e. the smoothing parameter)
##may strongly affect the result Here we use p-splines for each GEV
##parameter - so it's really CPU demanding but one can use 1 p-spline
##and 2 linear models.
##A simple linear model will be clearly faster...
loc.form <- y ~ rb(lat, knots = knots, degree = 3, penalty = .5)
scale.form <- y ~ rb(lat, knots = knots, degree = 3, penalty = .5)
shape.form <- y ~ rb(lon, knots = knots2, degree = 3, penalty = .5)

fitted <- fitmaxstab(data, locations, "gauss", loc.form, scale.form, shape.form,
                     control = list(ndeps = rep(1e-6, 24), trace = 10),
                     method = "BFGS")
fitted
op <- par(mfrow=c(1,3))
plot(locations[,2], param.loc, col = 2, ylim = c(7, 14),
     ylab = "location parameter", xlab = "latitude")
plot(fun, from = 0, to = 10, add = TRUE, col = 2)
points(locations[,2], predict(fitted)[,"loc"], col = "blue", pch = 5)
new.data <- cbind(lon = seq(0, 10, length = 100), lat = seq(0, 10, length = 100))
lines(new.data[,1], predict(fitted, new.data)[,"loc"], col = "blue")
legend("topleft", c("true values", "predict. values", "true curve", "predict. curve"),
       col = c("red", "blue", "red", "blue"), pch = c(1, 5, NA, NA), inset = 0.05,
       lty = c(0, 0, 1, 1), ncol = 2)

plot(locations[,2], param.scale, col = 2, ylim = c(7, 14),
     ylab = "scale parameter", xlab = "latitude")
plot(fun, from = 0, to = 10, add = TRUE, col = 2)
points(locations[,2], predict(fitted)[,"scale"], col = "blue", pch = 5)
lines(new.data[,1], predict(fitted, new.data)[,"scale"], col = "blue")
legend("topleft", c("true values", "predict. values", "true curve", "predict. curve"),
       col = c("red", "blue", "red", "blue"), pch = c(1, 5, NA, NA), inset = 0.05,
       lty = c(0, 0, 1, 1), ncol = 2)

plot(locations[,1], param.shape, col = 2,
     ylab = "shape parameter", xlab = "longitude")
plot(fun2, from = 0, to = 10, add = TRUE, col = 2)
points(locations[,1], predict(fitted)[,"shape"], col = "blue", pch = 5)
lines(new.data[,1], predict(fitted, new.data)[,"shape"], col = "blue")
legend("topleft", c("true values", "predict. values", "true curve", "predict. curve"),
       col = c("red", "blue", "red", "blue"), pch = c(1, 5, NA, NA), inset = 0.05,
       lty = c(0, 0, 1, 1), ncol = 2)
par(op)

## End(Not run)