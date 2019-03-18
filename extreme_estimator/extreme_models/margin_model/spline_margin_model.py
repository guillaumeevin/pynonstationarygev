
"""
Potentially, we could implement a spline model for the margin, to check results from Gaume

rbpspline

Fits a penalized spline with radial basis functions to data

Examplesn <- 200x <- runif(n)fun <- function(x) sin(3 * pi * x)y <- fun(x) + rnorm(n, 0, sqrt(0.4))
knots <- quantile(x, prob = 1:(n/4) / (n/4 + 1))fitted <- rbpspline(y, x, knots = knots, degree = 3)fittedplot(x, y)lines(fitted, col = 2)

"""