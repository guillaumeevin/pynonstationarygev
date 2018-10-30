library(stats4)
library(SpatialExtremes)

# Boolean for python call
call_main = !exists("python_wrapping")

if (call_main) {
    set.seed(42)
    N <- 50
    loc = 0; scale = 1; shape <- 1
    x_gev <- rgev(N, loc = loc, scale = scale, shape = shape)
    start_loc = 0; start_scale = 1; start_shape = 1
}

minus_log_likelihood_gev <- function(loc = 0, scale = 1, shape = 0){
     R = suppressWarnings(dgev(x_gev, loc = loc, scale = scale, shape = shape))
     -sum(log(R))
}

mle_gev <- function(loc, scale, shape){
    mle(minuslogl = minus_log_likelihood_gev, start = list(loc = loc, scale = scale, shape = shape))
}

main <- function(){
    res = mle_gev(start_loc, start_scale, start_shape)
    print(attributes(res))
    print(attr(res, 'coef'))
}

if (call_main) {
    main()
}