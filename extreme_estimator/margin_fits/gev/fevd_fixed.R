# Created on: 16/10/2019
library(extRemes)
library(stats4)
library(SpatialExtremes)

# TODO: send bug report on CRAN
fevd_fixed <- function (x, data, threshold = NULL, threshold.fun = ~1, location.fun = ~1,
    scale.fun = ~1, shape.fun = ~1, use.phi = FALSE, type = c("GEV",
        "GP", "PP", "Gumbel", "Exponential"), method = c("MLE",
        "GMLE", "Bayesian", "Lmoments"), initial = NULL, span,
    units = NULL, time.units = "days", period.basis = "year",
    na.action = na.fail, optim.args = NULL, priorFun = NULL,
    priorParams = NULL, proposalFun = NULL, proposalParams = NULL,
    iter = 9999, weights = 1, blocks = NULL, verbose = FALSE)
{
    if (verbose)
        begin.tiid <- Sys.time()
    out <- list()
    inout <- list()
    out$call <- match.call()
    if (!missing(data)) {
        out$data.name <- c(deparse(substitute(x)), deparse(substitute(data)))
    }
    else {
        out$data.name <- c(deparse(substitute(x)), "")
    }
    type <- match.arg(type)
    method <- match.arg(method)
    out$weights <- weights
    if (!missing(data)) {
        if (is.element(out$data.name[1], colnames(data))) {
            out$in.data <- TRUE
            wc <- out$data.name[1] == colnames(data)
            x <- c(data[, wc])
            x.fun <- ifelse(out$data.name[1] == "substitute(x)",
                deparse(x), out$data.name[1])
            x.fun <- formula(paste(x.fun, "~ 1"))
            out$x.fun <- x.fun
        }
        else if (is.formula(x)) {
            out$in.data <- TRUE
            x.fun <- x
            out$x.fun <- x.fun
            x <- model.response(model.frame(x.fun, data = data))
        }
        else out$in.data <- FALSE
        if (length(x) != nrow(data))
            stop("fevd: data must have same number of rows as the length of x.")
        if (!identical(weights, 1) && length(x) != length(weights))
            stop("fevd: weights should be the same length as x.")
        out$missing.values <- is.na(x)
        tmp <- cbind(x, data)
        tmp <- na.action(tmp)
        x <- tmp[, 1]
        data <- tmp[, -1, drop = FALSE]
    }
    else {
        out$missing.values <- is.na(x)
        if (is.formula(x))
            stop("fevd: Must provide data argument if you supply a formula to the x argument.")
        x <- na.action(x)
        out$in.data <- FALSE
    }
    if (!out$in.data) {
        data.pointer <- as.character(substitute(x))
        if (length(data.pointer) > 1)
            out$x <- x
        else out$data.pointer <- data.pointer
    }
    if (!is.null(blocks)) {
        if (type == "PP") {
            if (!is.element("nBlocks", names(blocks))) {
                if (is.element("data", names(blocks))) {
                  blocks$nBlocks <- nrow(blocks$data)
                }
                else stop("fevd: When supplying blocks, must provide 'blocks$nBlocks' if 'blocks$data' is  not provided.")
            }
            if (!is.element("weights", names(blocks)))
                blocks$weights <- 1
            if (!is.element("proportionMissing", names(blocks)))
                blocks$proportionMissing <- 0
            if (!is.element("threshold", names(blocks)) && !is.null(threshold)) {
                if (length(threshold) == 1) {
                  blocks$threshold <- threshold
                }
                else {
                  stop("fevd: No blocks$threshold specified and threshold is not constant. User must supply the threshold for each block via blocks$threshold.")
                }
            }
        }
        else {
            warning("fevd: Blocks are used only for type 'PP'; ignoring blocks argument.")
            blocks <- NULL
        }
    }
    cat('here1')
    out$x <- x
    if (!missing(data))
        out$cov.data <- data
    if (method == "MLE" && !is.null(priorFun))
        method <- "GMLE"
    else if (method == "GMLE" && is.null(priorFun)) {
        if (shape.fun != ~1)
            stop("fevd: must supply a prior function for GMLE method when shape parameter varies.")
        if (is.element(type, c("GEV", "GP", "PP"))) {
            priorFun <- "shapePriorBeta"
            if (is.null(priorParams))
                priorParams <- list(q = 6, p = 9)
        }
        else {
            warning("fevd: Using method MLE.  No default for specified arguments.")
            method <- "MLE"
        }
    }
    if (method == "GMLE") {
        out$priorFun <- priorFun
        out$priorParams <- priorParams
    }
    out$method <- method
    method <- tolower(method)
    out$type <- type
    type <- tolower(type)
    out$period.basis <- period.basis
    out$optim.args <- optim.args
    out$units <- units
    if (method == "bayesian" && missing(use.phi)) {
        use.phi <- TRUE
        if (verbose)
            cat("\n", "Setting use.phi argument to TRUE for greater stability in estimation (default with Bayesian method).  Use use.phi=FALSE if you prefer that.\n")
    }
    out$par.models <- list(threshold = threshold.fun, location = location.fun,
        scale = scale.fun, log.scale = use.phi, shape = shape.fun,
        term.names = list(threshold = all.vars(threshold.fun),
            location = all.vars(location.fun), scale = all.vars(scale.fun),
            shape = all.vars(shape.fun)))
    pars <- list()
    if (is.element(type, c("gp", "pp", "exponential", "beta",
        "pareto"))) {
        const.thresh <- check.constant(threshold.fun) & check.constant(threshold)
        out$const.thresh <- const.thresh
    }
    if (is.element(type, c("gev", "pp", "gumbel", "weibull",
        "frechet"))) {
        const.loc <- check.constant(location.fun)
        out$const.loc <- const.loc
    }
    const.scale <- check.constant(scale.fun)
    out$const.scale <- const.scale
    const.shape <- check.constant(shape.fun)
    out$const.shape <- const.shape
    if (is.element(type, c("pp", "gp", "exponential", "beta",
        "pareto"))) {
        if (missing(span)) {
            if (is.null(blocks)) {
                tiden <- attributes(x)$times
                n <- length(x)
                if (is.null(tiden)) {
                  tiden <- 1:n
                  start <- 1
                  end <- n
                  span <- end - start
                }
                else {
                  start <- tiden[1]
                  end <- tiden[n]
                  span <- as.numeric(difftime(as.POSIXlt(tiden)[n],
                    as.POSIXlt(tiden)[1], units = time.units))
                }
            }
            else {
                span <- blocks$nBlocks
            }
        }
        if (time.units == "days")
            npy <- 365.25
        else if (time.units == "months")
            npy <- 12
        else if (time.units == "years")
            npy <- 1
        else if (time.units == "hours")
            npy <- 24 * 365.25
        else if (time.units == "minutes")
            npy <- 60 * 24 * 365.25
        else if (time.units == "seconds")
            npy <- 60 * 60 * 24 * 365.25
        else {
            tmp.units <- unlist(strsplit(time.units, split = "/"))
            if (length(tmp.units) != 2)
                stop("fevd: invalid time.units argument.")
            numper <- as.numeric(tmp.units[1])
            if (is.na(numper))
                stop("fevd: invalid time.units argument.")
            pertiid <- tmp.units[2]
            if (!is.element(pertiid, c("day", "month", "year",
                "hour", "minute", "second")))
                stop("fevd: invalid time.units argument.")
            if (pertiid == "year")
                npy <- numper
            else if (pertiid == "month")
                npy <- numper * 12
            else if (pertiid == "day")
                npy <- numper * 365.25
            else if (pertiid == "hour")
                npy <- numper * 24 * 365.25
            else if (pertiid == "minute")
                npy <- numper * 60 * 24 * 365.25
            else if (pertiid == "second")
                npy <- numper * 60 * 60 * 24 * 365.25
        }
        if (!is.null(blocks))
            span <- span * npy
        out$time.units <- time.units
        out$span <- span/npy
        out$npy <- npy
        if (verbose)
            cat("\n", "Data span ", span/npy, "years", "\n")
    }
    else npy <- NULL
    n <- length(x)
    out$n <- n
    out$na.action <- deparse(substitute(na.action))
    if (!is.null(initial)) {
        if (!is.list(initial))
            stop("fevd: initial must be NULL or a list object.")
        find.init <- FALSE
        if (is.null(initial$location) && is.element(type, c("gev",
            "pp", "gumbel", "weibull", "frechet")))
            find.init <- TRUE
        if (use.phi && is.null(initial$log.scale))
            find.init <- TRUE
        if (!use.phi && is.null(initial$scale))
            find.init <- TRUE
        if (!is.element(type, c("gumbel", "exponential")) &&
            is.null(initial$shape))
            find.init <- TRUE
    }
    else {
        initial <- list()
        find.init <- TRUE
    }
    if (method != "lmoments") {
        if (verbose)
            cat("Setting up parameter model design matrices.\n")
        designs <- list()
        if (!missing(data)) {
            if (is.element(type, c("gp", "pp", "exponential",
                "beta", "pareto")))
                X.u <- setup.design(x = threshold.fun, data = data,
                  n = n, dname = "threshold.fun")
            if (is.element(type, c("gev", "pp", "gumbel", "weibull",
                "frechet"))) {
                X.loc <- setup.design(x = location.fun, data = data,
                  n = n, const = const.loc, dname = "location.fun")
                designs$X.loc <- X.loc
            }
            X.sc <- setup.design(x = scale.fun, data = data,
                n = n, const = const.scale, dname = "scale.fun")
            designs$X.sc <- X.sc
            if (!is.element(type, c("gumbel", "exponential"))) {
                X.sh <- setup.design(x = shape.fun, data = data,
                  n = n, const = const.shape, dname = "shape.fun")
                designs$X.sh <- X.sh
            }
        }
        else {
            if (is.element(type, c("gp", "pp", "exponential",
                "beta", "pareto")))
                X.u <- setup.design(x = threshold.fun, n = n,
                  dname = "threshold.fun")
            if (is.element(type, c("gev", "pp", "gumbel", "weibull",
                "frechet"))) {
                X.loc <- setup.design(x = location.fun, n = n,
                  const = const.loc, dname = "location.fun")
                designs$X.loc <- X.loc
            }
            X.sc <- setup.design(x = scale.fun, n = n, const = const.scale,
                dname = "scale.fun")
            designs$X.sc <- X.sc
            if (!is.element(type, c("gumbel", "exponential"))) {
                X.sh <- setup.design(x = shape.fun, n = n, const = const.shape,
                  dname = "shape.fun")
                designs$X.sh <- X.sh
            }
        }
        if (!is.null(blocks)) {
            blocks$designs <- list()
            if (is.element("data", names(blocks))) {
                blocks$X.u <- setup.design(x = threshold.fun,
                  data = blocks$data, n = blocks$nBlocks, dname = "threshold.fun")
                blocks$designs$X.loc <- setup.design(x = location.fun,
                  data = blocks$data, n = blocks$nBlocks, const = const.loc,
                  dname = "location.fun")
                blocks$designs$X.sc <- setup.design(x = scale.fun,
                  data = blocks$data, n = blocks$nBlocks, const = const.scale,
                  dname = "scale.fun")
                blocks$designs$X.sh <- setup.design(x = shape.fun,
                  data = blocks$data, n = blocks$nBlocks, const = const.shape,
                  dname = "shape.fun")
            }
            else {
                blocks$X.u <- setup.design(x = threshold.fun,
                  n = blocks$nBlocks, dname = "threshold.fun")
                blocks$designs$X.loc <- setup.design(x = location.fun,
                  n = blocks$nBlocks, const = const.loc, dname = "location.fun")
                blocks$designs$X.sc <- setup.design(x = scale.fun,
                  n = blocks$nBlocks, const = const.scale, dname = "scale.fun")
                blocks$designs$X.sh <- setup.design(x = shape.fun,
                  n = blocks$nBlocks, const = const.shape, dname = "shape.fun")
            }
        }
        if (verbose)
            cat("Parameter model design matrices set up.\n")
    }
    if (is.element(type, c("gp", "pp", "exponential", "beta",
        "pareto"))) {
        if (method != "lmoments")
            threshold <- rowSums(matrix(threshold, n, ncol(X.u),
                byrow = TRUE) * X.u)
        if (!is.null(blocks))
            blocks$threshold <- rowSums(matrix(blocks$threshold,
                blocks$nBlocks, ncol(blocks$X.u), byrow = TRUE) *
                blocks$X.u)
        excess.id <- x > threshold
        if (all(threshold == threshold[1]))
            out$threshold <- threshold[1]
        else out$threshold <- threshold
        out$rate <- mean(excess.id)
    }
    out$blocks <- blocks
    if (method == "lmoments" || find.init) {
        if (method == "lmoments") {
            if (verbose)
                cat("Beginning estimation procedure.\n")
            is.constant <- unlist(lapply(list(u = threshold,
                loc = location.fun, scale = scale.fun, sh = shape.fun),
                check.constant))
            if (!all(is.constant))
                warning("fevd: For method Lmoments, this function does not handle covariates in the parameters.  Fitting w/o covariates.")
            if (!is.element(type, c("gev", "gp")))
                stop("fevd: currently, Lmoments are only handled for estimation of GEV and GP distribution parameters.")
        }
        xtemp <- x
        class(xtemp) <- "lmoments"
        ipars1 <- try(initializer(xtemp, model = type, threshold = threshold,
            npy = npy, blocks = blocks), silent = TRUE)
        if (class(ipars1) != "try-error") {
            if (ipars1["scale"] <= 0)
                ipars1["scale"] <- 1e-08
            if (method == "lmoments") {
                out$results <- ipars1
                class(out) <- "fevd"
                return(out)
            }
        }
        else {
            ipars1 <- NULL
            if (method == "lmoments")
                stop("fevd: Sorry, could not find L-moments estimates.")
        }
    }
    if ((method != "lmoments") && find.init) {
        xtemp <- x
        class(xtemp) <- "moms"
        ipars2 <- try(initializer(xtemp, model = type, threshold = threshold,
            npy = npy, blocks = blocks), silent = TRUE)
        if (class(ipars2) != "try-error") {
            if (ipars2["scale"] <= 0)
                ipars2["scale"] <- 1e-08
        }
        else ipars2 <- NULL
        if (!is.element(type, c("pp", "gp", "exponential", "beta",
            "pareto", "gumbel"))) {
            if (!is.null(ipars1))
                testLmoments <- levd(x, location = ipars1["location"],
                  scale = ipars1["scale"], shape = ipars1["shape"],
                  type = out$type, npy = npy)
            else testLmoments <- Inf
            if (!is.null(ipars2))
                testMoments <- levd(x, location = ipars2["location"],
                  scale = ipars2["scale"], shape = ipars2["shape"],
                  type = out$type, npy = npy)
            else testMoments <- Inf
        }
        else if (type == "pp") {
            if (!is.null(ipars1)) {
                if (!is.null(blocks)) {
                  blocks$location = ipars1["location"]
                  blocks$scale = ipars1["scale"]
                  blocks$shape = ipars1["shape"]
                }
                testLmoments <- levd(x, threshold = threshold,
                  location = ipars1["location"], scale = ipars1["scale"],
                  shape = ipars1["shape"], type = out$type, npy = npy,
                  blocks = blocks)
            }
            else testLmoments <- Inf
            if (!is.null(ipars2)) {
                if (!is.null(blocks)) {
                  blocks$location = ipars2["location"]
                  blocks$scale = ipars2["scale"]
                  blocks$shape = ipars2["shape"]
                }
                testMoments <- levd(x, threshold = threshold,
                  location = ipars2["location"], scale = ipars2["scale"],
                  shape = ipars2["shape"], type = out$type, npy = npy,
                  blocks = blocks)
            }
            else testMoments <- Inf
            if (!is.null(blocks))
                blocks$location <- blocks$scale <- blocks$shape <- NULL
        }
        else if (!is.element(type, c("gumbel", "exponential"))) {
            if (!is.null(ipars1))
                testLmoments <- levd(x, threshold = threshold,
                  scale = ipars1["scale"], shape = ipars1["shape"],
                  type = out$type, npy = npy)
            else testLmoments <- Inf
            if (!is.null(ipars2))
                testMoments <- levd(x, threshold = threshold,
                  scale = ipars2["scale"], shape = ipars2["shape"],
                  type = out$type, npy = npy)
            else testMoments <- Inf
        }
        else if (type == "gumbel") {
            if (!is.null(ipars1))
                testLmoments <- levd(x, location = ipars1["location"],
                  scale = ipars1["scale"], type = out$type, npy = npy)
            else testLmoments <- Inf
            if (!is.null(ipars2))
                testMoments <- levd(x, location = ipars2["location"],
                  scale = ipars2["scale"], type = out$type, npy = npy)
            else testMoments <- Inf
        }
        else if (type == "exponential") {
            if (!is.null(ipars1))
                testLmoments <- levd(x, threshold = threshold,
                  scale = ipars1["scale"], shape = 0, type = out$type,
                  npy = npy)
            else testLmoments <- Inf
            if (!is.null(ipars2))
                testMoments <- levd(x, threshold = threshold,
                  scale = ipars2["scale"], shape = 0, type = out$type,
                  npy = npy)
            else testMoments <- Inf
        }
        if (is.finite(testLmoments) || is.finite(testMoments)) {
            if (testLmoments < testMoments) {
                if (is.null(initial$location) && !is.element(type,
                  c("gp", "exponential", "beta", "pareto")))
                  initial$location <- ipars1["location"]
                if (is.null(initial$log.scale) && use.phi)
                  initial$log.scale <- log(ipars1["scale"])
                else if (is.null(initial$scale))
                  initial$scale <- ipars1["scale"]
                if (!is.element(type, c("gumbel", "exponential")) &&
                  is.null(initial$shape))
                  initial$shape <- ipars1["shape"]
                if (verbose)
                  cat("Using Lmoments estimates as initial estimates.  Initial value =",
                    testLmoments, "\n")
            }
            else {
                if (is.null(initial$location) && !is.element(type,
                  c("gp", "exponential", "beta", "pareto")))
                  initial$location <- ipars2["location"]
                if (is.null(initial$log.scale) && use.phi)
                  initial$log.scale <- log(ipars2["scale"])
                else if (is.null(initial$scale))
                  initial$scale <- ipars2["scale"]
                if (!is.element(type, c("gumbel", "exponential")) &&
                  is.null(initial$shape))
                  initial$shape <- ipars2["shape"]
                if (verbose)
                  cat("Initial estimates found where necessary (not from Lmoments).  Initial value =",
                    testMoments, "\n")
            }
        }
        else {
            if (is.null(initial$location) && !is.element(type,
                c("gp", "exponential", "beta", "pareto")))
                initial$location <- 0
            if (is.null(initial$log.scale) && use.phi)
                initial$log.scale <- 0
            else if (is.null(initial$scale))
                initial$scale <- 1
            if (!is.element(type, c("gumbel", "exponential")) &&
                is.null(initial$shape))
                initial$shape <- 0.01
            warning("fevd: L-moments and Moment initial parameter estimates could not be calculated.  Using arbitrary starting values.")
        }
        inout <- list(Lmoments = list(pars = ipars1, likelihood = testLmoments),
            MOM = list(pars = ipars2, likelihood = testMoments))
    }
    if (!is.null(initial$location)) {
        if (ncol(X.loc) != length(initial$location)) {
            if ((length(initial$location) == 1) && ncol(X.loc) >
                1)
                initial$location <- c(initial$location, rep(0,
                  ncol(X.loc) - 1))
            else stop("fevd: initial parameter estimates must have length 1 or number of parameters present.  Incorrect number for location parameter.")
        }
        if (length(initial$location) == 1)
            names(initial$location) <- "location"
        else names(initial$location) <- paste("mu", 0:(ncol(X.loc) -
            1), sep = "")
    }
    if (use.phi && (ncol(X.sc) != length(initial$log.scale))) {
        if ((length(initial$log.scale) == 1) && ncol(X.sc) >
            1)
            initial$log.scale <- c(initial$log.scale, rep(0,
                ncol(X.sc) - 1))
        else stop("fevd: initial parameter estimates must have length 1 or number of parameters present.  Incorrect number for log(scale) parameter.")
    }
    else if (!use.phi && (ncol(X.sc) != length(initial$scale))) {
        if ((length(initial$scale) == 1) && ncol(X.sc) > 1)
            initial$scale <- c(initial$scale, rep(0, ncol(X.sc) -
                1))
        else stop("fevd: initial parameter estimates must have length 1 or number of parameters present.  Incorrect number for scale parameter.")
    }
    if (use.phi) {
        if (length(initial$log.scale) == 1)
            names(initial$log.scale) <- "log.scale"
        else names(initial$log.scale) <- paste("phi", 0:(ncol(X.sc) -
            1), sep = "")
    }
    else {
        if (length(initial$scale) == 1)
            names(initial$scale) <- "scale"
        else names(initial$scale) <- paste("sigma", 0:(ncol(X.sc) -
            1), sep = "")
    }
    if (!is.element(type, c("gumbel", "exponential"))) {
        if (ncol(X.sh) != length(initial$shape)) {
            if ((length(initial$shape) == 1) && ncol(X.sh) >
                1)
                initial$shape <- c(initial$shape, rep(0, ncol(X.sh) -
                  1))
            else stop("fevd: initial parameter estimates must have length 1 or number of parameters present.  Incorrect number for shape parameter.")
        }
        if (length(initial$shape) == 1)
            names(initial$shape) <- "shape"
        else names(initial$shape) <- paste("xi", 0:(ncol(X.sh) -
            1), sep = "")
    }
    if (is.element(method, c("mle", "gmle"))) {
        if (use.phi)
            init.pars <- c(initial$location, initial$log.scale,
                initial$shape)
        else init.pars <- c(initial$location, initial$scale,
            initial$shape)
        if (type == "exponential" && const.scale) {
            if (method == "gmle")
                warning("Method MLE used.")
            res <- list()
            excess.id <- x > threshold
            mle <- mean(x[excess.id] - threshold[excess.id])
            names(mle) <- "scale"
            res$par <- mle
            k <- sum(excess.id)
            res$n <- k
            res$value <- k * (log(mle) + 1)
        }
        else {
            if (!is.null(a <- optim.args)) {
                anam <- names(a)
                if (!is.element("gr", anam)) {
                  if (method == "mle")
                    opt.gr <- grlevd
                  else opt.gr <- NULL
                }
                else opt.gr <- a$gr
                if (is.null(a$method) && use.phi)
                  opt.method <- ifelse(is.element(type, c("gev",
                    "gp", "pp", "gumbel")), "BFGS", "L-BFGS-B")
                else opt.method <- a$method
                if (is.element(type, c("weibull", "beta", "frechet",
                  "pareto")))
                  opt.method <- "L-BFGS-B"
                if (is.element(opt.method, c("L-BFGS-B", "Brent"))) {
                  if (is.null(a$lower)) {
                    if (!is.element(type, c("frechet", "pareto")))
                      opt.lower <- -Inf
                    else opt.lower <- c(rep(-Inf, length(init.pars) -
                      1), 0)
                  }
                  else {
                    if (is.element(type, c("frechet", "pareto")))
                      opt.lower <- c(a$lower[1:(length(init.pars) -
                        1)], 0)
                    else opt.lower <- a$lower
                  }
                  if (is.null(a$upper)) {
                    if (!is.element(type, c("weibull", "beta")))
                      opt.upper <- Inf
                    else opt.upper <- c(rep(Inf, length(init.pars) -
                      1), 0)
                  }
                  else {
                    if (is.element(type, c("weibull", "beta")))
                      opt.upper <- c(a$upper[1:(length(init.pars) -
                        1)], 0)
                    else opt.upper <- a$upper
                  }
                }
                else {
                  opt.lower <- -Inf
                  opt.upper <- Inf
                }
                if (is.null(a$control))
                  opt.control <- list()
                else opt.control <- a$control
                anam <- names(a$control)
                if (!is.element("trace", anam) && verbose)
                  opt.control$trace <- 6
                if (is.null(a$hessian))
                  opt.hessian <- TRUE
                else opt.hessian <- a$hessian
            }
            else {
                if (method == "mle")
                  opt.gr <- grlevd
                else opt.gr <- NULL
                if (is.element(type, c("gev", "gp", "pp", "gumbel")))
                  opt.method <- "BFGS"
                else opt.method <- "L-BFGS-B"
                if (!is.element(type, c("frechet", "pareto")))
                  opt.lower <- -Inf
                else opt.lower <- c(rep(-Inf, length(init.pars) -
                  1), 0)
                if (!is.element(type, c("weibull", "beta")))
                  opt.upper <- Inf
                else opt.upper <- c(rep(Inf, length(init.pars) -
                  1), 0)
                if (verbose)
                  opt.control <- list(trace = 6)
                else opt.control <- list()
                opt.hessian <- TRUE
            }
            parnames <- names(init.pars)
            out$parnames <- parnames
            if (verbose && (method != "lmoments")) {
                cat("Initial estimates are:\n")
                print(init.pars)
                cat("Beginning estimation procedure.\n")
            }
            if (type == "pp" && find.init) {
                if (verbose)
                  cat("\n", "First fitting a GP-Poisson model in order to try to get a good initial estimate as PP likelihoods can be very unstable.\n")
                look <- out
                look$type <- "GP"
                des2 <- designs
                des2$X.loc <- NULL
                if (!missing(data))
                  resGP <- optim(init.pars[-(1:ncol(X.loc))],
                    oevd, gr = opt.gr, o = look, des = des2,
                    x = x, data = data, u = threshold, npy = npy,
                    phi = use.phi, method = opt.method, lower = opt.lower,
                    upper = opt.upper, control = opt.control,
                    hessian = opt.hessian)
                else resGP <- optim(init.pars[-(1:ncol(X.loc))],
                  oevd, gr = opt.gr, o = look, des = des2, x = x,
                  u = threshold, npy = npy, phi = use.phi, method = opt.method,
                  lower = opt.lower, upper = opt.upper, control = opt.control,
                  hessian = opt.hessian)
                tmpi <- resGP$par
                if (is.null(blocks)) {
                  lrate <- npy * mean(x > threshold)
                }
                else {
                  lrate <- sum(x > threshold)/(blocks$nBlocks *
                    mean(blocks$weights))
                }
                xi3 <- tmpi[(ncol(X.sc) + 1):length(tmpi)]
                if (!use.phi)
                  sigma3 <- exp(tmpi[1:ncol(X.sc)] + xi3 * log(lrate))
                else sigma3 <- tmpi[1:ncol(X.sc)] + xi3 * log(lrate)
                lp <- lrate^(-xi3) - 1
                if (all(is.finite(lp)))
                  mu3 <- mean(threshold) - (sigma3/xi3) * lp
                else mu3 <- mean(x)
                nloc <- ncol(X.loc)
                if (length(mu3) < nloc)
                  mu3 <- c(mu3, rep(0, nloc - length(mu3)))
                else mu3 <- mu3[1]
                if (!is.null(blocks)) {
                  blocks$location <- rowSums(matrix(mu3, blocks$nBlocks,
                    nloc) * blocks$designs$X.loc)
                  blocks$scale = rowSums(matrix(sigma3, blocks$nBlocks,
                    ncol(blocks$designs$X.sc)) * blocks$designs$X.sc)
                  blocks$shape = rowSums(matrix(xi3, blocks$nBlocks,
                    ncol(blocks$designs$X.sh)) * blocks$designs$X.sh)
                }
                if (all(is.finite(c(mu3, sigma3, xi3)))) {
                  testGPmle <- try(levd(x = x, threshold = threshold,
                    location = rowSums(matrix(mu3, n, nloc) *
                      X.loc), scale = rowSums(matrix(sigma3,
                      n, ncol(X.sc)) * X.sc), shape = rowSums(matrix(xi3,
                      n, ncol(X.sh)) * X.sh), type = "PP", npy = npy,
                    blocks = blocks), silent = TRUE)
                  if (class(testGPmle) == "try-error")
                    testGPmle <- Inf
                }
                else testGPmle <- Inf
                if (!is.null(blocks))
                  blocks$location <- blocks$scale <- blocks$shape <- NULL
                if (is.finite(testLmoments) || is.finite(testMoments) ||
                  is.finite(testGPmle)) {
                  if ((testGPmle < testLmoments) && (testGPmle <
                    testMoments)) {
                    if (verbose)
                      cat("\n", "Changing initial estimates to those based on GP MLEs.  They are: \n")
                    if (use.phi)
                      init.pars <- c(mu3, log(sigma3), xi3)
                    else init.pars <- c(mu3, sigma3, xi3)
                    names(init.pars) <- parnames
                    if (verbose)
                      print(init.pars)
                  }
                  else if (verbose)
                    cat("\n", "Sticking with originally chosen initial estimates.\n")
                }
                inout$PoissonGP <- list(pars = c(mu3, sigma3,
                  xi3), likelihood = testGPmle)
            }
            if (method == "mle") {
                if (!missing(data)) {
                  res <- optim(init.pars, oevd, gr = opt.gr,
                    o = out, des = designs, x = x, data = data,
                    u = threshold, span = span/npy, npy = npy,
                    phi = use.phi, blocks = blocks, method = opt.method,
                    lower = opt.lower, upper = opt.upper, control = opt.control,
                    hessian = opt.hessian)
                }
                else {
                  res <- optim(init.pars, oevd, gr = opt.gr,
                    o = out, des = designs, x = x, u = threshold,
                    span = span/npy, npy = npy, phi = use.phi,
                    blocks = blocks, method = opt.method, lower = opt.lower,
                    upper = opt.upper, control = opt.control,
                    hessian = opt.hessian)
                }
            }
            else if (method == "gmle") {
                if (!missing(data)) {
                  res <- optim(init.pars, oevdgen, gr = opt.gr,
                    o = out, des = designs, x = x, data = data,
                    u = threshold, span = span/npy, npy = npy,
                    phi = use.phi, blocks = blocks, priorFun = priorFun,
                    priorParams = priorParams, method = opt.method,
                    lower = opt.lower, upper = opt.upper, control = opt.control,
                    hessian = opt.hessian)
                }
                else {
                  res <- optim(init.pars, oevdgen, gr = opt.gr,
                    o = out, des = designs, x = x, u = threshold,
                    span = span/npy, npy = npy, phi = use.phi,
                    blocks = blocks, priorFun = priorFun, priorParams = priorParams,
                    method = opt.method, lower = opt.lower, upper = opt.upper,
                    control = opt.control, hessian = opt.hessian)
                }
            }
        }
        if (is.element("shape", names(res$par))) {
            if (is.element(type, c("frechet", "pareto"))) {
                res$par["shape"] <- abs(res$par["shape"])
                if (res$par["shape"] == 0) {
                  warning("fevd: shape parameter estimated to be zero.  Re-setting to be 1e16.")
                  res$par["shape"] <- 1e+16
                }
            }
            else {
                if (is.element(type, c("weibull", "beta")))
                  res$par["shape"] <- -abs(res$par["shape"])
                if (res$par["shape"] == 0) {
                  warning("fevd: shape parameter estimated to be zero.  Re-setting to be -1e16.")
                  res$par["shape"] <- -1e+16
                }
            }
        }
        res$num.pars <- list(location = ncol(designs$X.loc),
            scale = ncol(designs$X.sc), shape = ncol(designs$X.sh))
        out$results <- res
    }
    else if (method == "bayesian") {
        if (is.element(type, c("gev", "gumbel", "weibull", "frechet",
            "pp"))) {
            nloc <- ncol(X.loc)
            loc.names <- names(initial$location)
        }
        else {
            nloc <- 0
            loc.names <- NULL
        }
        nsc <- ncol(X.sc)
        if (use.phi && is.null(initial$log.scale)) {
            initial$log.scale <- log(initial$scale)
            if (nsc == 1)
                names(initial$log.scale) <- "log.scale"
            else names(initial$log.scale) <- paste("phi", 0:(nsc -
                1), sep = "")
        }
        # The 6 following lines correspond to the bug fix. The case use.phi = FALSE was not handle properly before
        if (use.phi){
            sc.names = names(initial$log.scale)
        }
        else {
            sc.names = names(initial$scale)
        }
        if (!is.element(type, c("gumbel", "exponential"))) {
            nsh <- ncol(X.sh)
            sh.names <- names(initial$shape)
        }
        else {
            nsh <- 0
            sh.names <- NULL
        }
        np <- nloc + nsc + nsh
        find.priorParams <- FALSE
        if (is.null(priorFun) && is.null(priorParams))
            find.priorParams <- TRUE
        else if (is.null(priorFun) && (is.null(priorParams$m) ||
            is.null(priorParams$v)))
            find.priorParams <- TRUE
        else if (!is.null(priorFun)) {
            if (priorFun == "fevdPriorDefault") {
                if (is.null(priorParams))
                  find.priorParams <- TRUE
                else if (is.null(priorParams$m) || is.null(priorParams$v))
                  find.priorParams <- TRUE
            }
        }
        if (is.null(priorFun) || find.priorParams) {
            if (is.null(priorFun))
                priorFun <- "fevdPriorDefault"
            if (find.priorParams) {
                xtemp <- x
                class(xtemp) <- "mle"
                if (verbose)
                  cat("\n", "Finding MLE to obtain prior means and variances.\n")
                if (missing(data)) {
                  if (missing(span))
                    hold <- initializer(xtemp, u = threshold,
                      use.phi = use.phi, type = out$type, time.units = time.units,
                      period.basis = period.basis, blocks = blocks)
                  else hold <- initializer(xtemp, u = threshold,
                    use.phi = use.phi, type = out$type, span = span,
                    time.units = time.units, period.basis = period.basis,
                    blocks = blocks)
                }
                else {
                  if (missing(span))
                    hold <- initializer(xtemp, data = data, u = threshold,
                      u.fun = threshold.fun, loc.fun = location.fun,
                      sc.fun = scale.fun, sh.fun = shape.fun,
                      use.phi = use.phi, type = out$type, time.units = time.units,
                      period.basis = period.basis, blocks = blocks)
                  else hold <- initializer(xtemp, data = data,
                    u = threshold, u.fun = threshold.fun, loc.fun = location.fun,
                    sc.fun = scale.fun, sh.fun = shape.fun, use.phi = use.phi,
                    type = out$type, span = span, time.units = time.units,
                    period.basis = period.basis, blocks = blocks)
                }
                if (is.null(priorParams))
                  priorParams <- list(m = hold[1:np], v = rep(10,
                    np))
                else if (is.null(priorParams$m))
                  priorParams$m <- hold[1:np]
                else if (is.null(priorParams$v))
                  priorParams$v <- rep(10, np)
            }
        }
        out$priorFun <- priorFun
        out$priorParams <- priorParams
        if (is.null(proposalFun)) {
            proposalFun <- "fevdProposalDefault"
            if (is.null(proposalParams))
                proposalParams <- list(sd = rep(0.1, np))
        }
        out$proposalFun <- proposalFun
        out$proposalParams <- proposalParams
        chain.info <- matrix(NA, iter, np + 2)
        print(dim(chain.info))
        print(c(loc.names, sc.names, sh.names, "loglik", "prior"))
        colnames(chain.info) <- c(loc.names, sc.names, sh.names,
            "loglik", "prior")
        chain.info[2:iter, 1:np] <- 0
        res <- matrix(NA, iter, np + 1)
        res[, np + 1] <- 0
        colnames(res) <- c(loc.names, sc.names, sh.names, "new")
        if (nloc > 0)
            res[1, 1:nloc] <- initial$location
        if (use.phi)
            res[1, (nloc + 1):(nloc + nsc)] <- initial$log.scale
        else res[1, (nloc + 1):(nloc + nsc)] <- initial$scale
        if (type != "Gumbel")
            res[1, (nloc + nsc + 1):np] <- initial$shape
        theta.i <- res[1, 1:np]
        if (verbose) {
            cat("\n", "Finding log-Likelihood of initial parameter values:\n")
            print(theta.i)
        }
        if (!missing(data))
            ll.i <- -oevd(p = res[1, ], o = out, des = designs,
                x = x, data = data, u = threshold, span = span,
                npy = npy, phi = use.phi, blocks = blocks)
        else ll.i <- -oevd(p = res[1, ], o = out, des = designs,
            x = x, u = threshold, span = span, npy = npy, phi = use.phi,
            blocks = blocks)
        if (verbose)
            cat("\n", "Finding prior df value of initial parameter values.\n")
        p.i <- do.call(priorFun, c(list(theta = theta.i), priorParams))
        chain.info[1, np + 1] <- ll.i
        chain.info[1, np + 2] <- p.i
        if (verbose)
            cat("\n", "Beginning the MCMC iterations (", iter,
                " total iterations)\n")
        for (i in 2:iter) {
            if (verbose && i <= 10)
                cat(i, " ")
            if (verbose && i%%100 == 0)
                cat(i, " ")
            ord <- sample(1:np, np)
            theta.star <- theta.i
            acc <- 0
            for (j in ord) {
                par.star <- do.call(proposalFun, c(list(p = theta.i,
                  ind = j), proposalParams))
                theta.star[j] <- par.star[j]
                if (!missing(data))
                  ll.star <- -oevd(p = theta.star, o = out, des = designs,
                    x = x, data = data, u = threshold, span = span,
                    npy = npy, phi = use.phi, blocks = blocks)
                else ll.star <- -oevd(p = theta.star, o = out,
                  des = designs, x = x, u = threshold, span = span,
                  npy = npy, phi = use.phi, blocks = blocks)
                prior.star <- do.call(priorFun, c(list(theta = theta.star),
                  priorParams))
                look <- will.accept(ll.i = ll.i, prior.i = p.i,
                  ll.star = ll.star, prior.star = prior.star,
                  log = TRUE)
                if (look$accept) {
                  p.i <- prior.star
                  ll.i <- ll.star
                  theta.i <- theta.star
                  acc <- acc + 1
                  chain.info[i, j] <- 1
                }
            }
            res[i, ] <- c(theta.i, acc)
            chain.info[i, (np + 1):(np + 2)] <- c(ll.i, p.i)
        }
        if (verbose)
            cat("\n", "Finished MCMC iterations.\n")
        out$results <- res
        out$chain.info <- chain.info
    }
    else stop("fevd: invalid method argument.")
    out$initial.results <- inout
    if (verbose)
        print(Sys.time() - begin.tiid)
    if (method == "GMLE")
        cl <- "mle"
    else cl <- tolower(method)
    class(out) <- "fevd"
    return(out)
}

