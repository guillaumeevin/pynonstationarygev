# Title     : TODO
# Objective : TODO
# Created by: erwan
# Created on: 31/10/2019

ci.fevd.mle_fixed <- function (x, alpha = 0.05, type = c("return.level", "parameter"),
    return.period = 100, which.par = 1, R = 502, method = c("normal",
        "boot", "proflik"), xrange = NULL, nint = 20, verbose = FALSE,
    tscale = FALSE, return.samples = FALSE, ...)
{
    if (missing(method))
        miss.meth <- TRUE
    else miss.meth <- FALSE
    method <- tolower(method)
    method <- match.arg(method)
    type <- tolower(type)
    type <- match.arg(type)
    theta.hat <- x$results$par
    theta.names <- names(theta.hat)
    np <- length(theta.hat)
    if (type == "parameter" && missing(which.par))
        which.par <- 1:np
    if (any(theta.names == "log.scale")) {
        id <- theta.names == "log.scale"
        theta.hat[id] <- exp(theta.hat[id])
        theta.names[id] <- "scale"
        names(theta.hat) <- theta.names
    }
    const <- is.fixedfevd(x)
    if (type == "return.level")
        par.name <- paste(return.period, "-", x$period.basis,
            " return level", sep = "")
    else if (type == "parameter")
        par.name <- theta.names[which.par]
    if (type == "return.level" && !const) {
        return(ci.rl.ns.fevd.mle(x = x, alpha = alpha, return.period = return.period,
            method = method, verbose = verbose, return.samples = return.samples,
            ...))
    }
    if (type == "parameter")
        p <- theta.hat[which.par]
    else {
        if (is.element(x$type, c("PP", "GP", "Beta", "Pareto",
            "Exponential")))
            lam <- mean(c(datagrabber(x)[, 1]) > x$threshold)
        else lam <- 1
        if (is.element(x$type, c("PP", "GEV", "Gumbel", "Weibull",
            "Frechet")))
            loc <- theta.hat["location"]
        else loc <- 0
        scale <- theta.hat["scale"]
        if (!is.element(x$type, c("Gumbel", "Exponential")))
            shape <- theta.hat["shape"]
        else shape <- 0
        if (x$type == "PP")
            mod <- "GEV"
        else mod <- x$type
        p <- rlevd(period = return.period, loc = loc, scale = scale,
            shape = shape, threshold = x$threshold, type = mod,
            npy = x$npy, rate = lam)
    }
    if (verbose) {
        cat("\n", "Preparing to calculate ", (1 - alpha) * 100,
            "% CI for ", ifelse(type == "return.level", paste(return.period,
                "-", x$period.basis, " return level", sep = ""),
                paste(par.name, " parameter", sep = "")), "\n")
        cat("\n", "Model is ", ifelse(const, " fixed", " non-stationary."),
            "\n")
        if (method == "normal")
            cat("\n", "Using Normal Approximation Method.\n")
        else if (method == "boot")
            cat("\n", "Using Bootstrap Method.\n")
        else if (method == "proflik")
            cat("\n", "Using Profile Likelihood Method.\n")
    }
    if (method == "normal") {
        method.name <- "Normal Approx."
        z.alpha <- qnorm(alpha/2, lower.tail = FALSE)
        cov.theta <- parcov.fevd(x)
        if (is.null(cov.theta))
            stop("ci: Sorry, unable to calculate the parameter covariance matrix.  Maybe try a different method.")
        var.theta <- diag(cov.theta)
        if (any(var.theta < 0))
            stop("ci: negative Std. Err. estimates obtained.  Not trusting any of them.")
        if (type == "parameter") {
            se.theta <- sqrt(var.theta)
            if (tscale) {
                if (!const && !is.element("scale", theta.names) &&
                  !is.element("shape", theta.names) && !all(x$threshold ==
                  x$threshold[1])) {
                  stop("ci: invalid argument configurations.")
                }
                if (!is.element(x$type, c("GP", "Beta", "Pareto")))
                  stop("ci: invalid argument configurations.")
                theta.hat["scale"] <- theta.hat["scale"] - theta.hat["shape"] *
                  x$threshold
                theta.names[theta.names == "scale"] <- "tscale"
                if (!any(theta.names[which.par] == "tscale"))
                  stop("ci: invalid argument configurations.")
                names(theta.hat) <- theta.names
                p <- theta.hat[which.par]
                d <- rbind(1, -x$threshold)
                names(se.theta) <- theta.names
                se.theta["tscale"] <- sqrt(t(d) %*% cov.theta %*%
                  d)
            }
            else se.theta <- sqrt(var.theta)[which.par]
            se.theta <- se.theta[which.par]
            par.name <- theta.names[which.par]
        }
        else if (type == "return.level") {
            grads <- rlgrad.fevd(x, period = return.period)
            grads <- t(grads)
            if (is.element(x$type, c("GP", "Beta", "Pareto",
                "Exponential"))) {
                if (x$type == "Exponential")
                  cov.theta <- diag(c(lam * (1 - lam)/x$n, var.theta))
                else cov.theta <- rbind(c(lam * (1 - lam)/x$n,
                  0, 0), cbind(0, cov.theta))
            }
            else lam <- 1
            var.theta <- t(grads) %*% cov.theta %*% grads
        }
        else stop("ci: invalid type argument.  Must be return.level or parameter.")
        if (length(p) > 1) {
            if (type == "return.level")
                se.theta <- sqrt(diag(var.theta))
            out <- cbind(p - z.alpha * se.theta, p, p + z.alpha *
                se.theta)
            rownames(out) <- par.name
            conf.level <- paste(round((1 - alpha) * 100, digits = 2),
                "%", sep = "")
            colnames(out) <- c(paste(conf.level, " lower CI",
                sep = ""), "Estimate", paste(conf.level, " upper CI",
                sep = ""))
            attr(out, "data.name") <- x$call
            attr(out, "method") <- method.name
            attr(out, "conf.level") <- (1 - alpha) * 100
            class(out) <- "ci"
            return(out)
        }
        else out <- c(p - z.alpha * sqrt(var.theta[which.par]),
            p, p + z.alpha * sqrt(var.theta[which.par]))
    }
    else if (method == "boot") {
        method.name <- "Parametric Bootstrap"
        if (verbose)
            cat("\n", "Simulating data from fitted model.  Size = ",
                R, "\n")
        if (const) {
            if (is.null(x$blocks)) {
                Z <- rextRemes(x, n = R * x$n)
                Z <- matrix(Z, x$n, R)
            }
            else {
                Z <- rextRemes(x, n = round(R * x$npy * x$blocks$nBlocks))
                Z <- matrix(Z, round(x$npy * x$blocks$nBlocks),
                  R)
            }
        }
        else Z <- rextRemes(x, n = R)
        if (verbose)
            cat("\n", "Simulated data found.\n")
        y <- datagrabber(x, response = FALSE)
        if (is.element(x$type, c("PP", "GP", "Exponential", "Beta",
            "Pareto"))) {
            x2 <- datagrabber(x, cov.data = FALSE)
            eid <- x2 > x$threshold
            Z2 <- matrix(x$threshold, x$n, R)
            Z2[eid, ] <- Z[eid, ]
            Z <- Z2
            lam <- mean(eid)
        }
        else {
            eid <- !logical(x$n)
            lam <- 1
        }
        ipar <- list()
        if (any(is.element(c("location", "mu0"), theta.names))) {
            if (is.element("location", theta.names))
                ipar$location <- theta.hat["location"]
            else {
                id <- substring(theta.names, 1, 2) == "mu"
                ipar$location <- theta.hat[id]
            }
        }
        if (is.element("scale", theta.names))
            ipar$scale <- theta.hat["scale"]
        else {
            if (!x$par.models$log.scale)
                id <- substring(theta.names, 1, 3) == "sig"
            else id <- substring(theta.names, 1, 3) == "phi"
            ipar$scale <- theta.hat[id]
        }
        if (!is.element(x$type, c("Gumbel", "Exponential"))) {
            if (is.element("shape", theta.names))
                ipar$shape <- theta.hat["shape"]
            else {
                id <- substring(theta.names, 1, 2) == "xi"
                ipar$shape <- theta.hat[id]
            }
        }
        bfun <- function(z, x, y, p, ipar, eid, rate) {
            pm <- x$par.models
            if (is.null(y))
                fit <- fevd(x = z, threshold = x$threshold, location.fun = pm$location,
                  scale.fun = pm$scale, shape.fun = pm$shape,
                  use.phi = pm$log.scale, type = x$type, method = x$method,
                  initial = ipar, span = x$span, time.units = x$time.units,
                  period.basis = x$period.basis, optim.args = x$optim.args,
                  priorFun = x$priorFun, priorParams = x$priorParams,
                  verbose = FALSE)
            else fit <- fevd(x = z, data = y, threshold = x$threshold,
                location.fun = pm$location, scale.fun = pm$scale,
                shape.fun = pm$shape, use.phi = pm$log.scale,
                type = x$type, method = x$method, initial = ipar,
                span = x$span, time.units = x$time.units, period.basis = x$period.basis,
                optim.args = x$optim.args, priorFun = x$priorFun,
                priorParams = x$priorParams, verbose = FALSE)
            fit$cov.data <- y
            res <- distill(fit, cov = FALSE)
            return(res)
        }
        if (verbose)
            cat("\n", "Fitting model to simulated data sets (this may take a while!).")
        if (type == "parameter") {
            sam <- apply(Z, 2, bfun, x = x, y = y, ipar = ipar)
            if (tscale) {
                if (!const && !is.element("scale", theta.names) &&
                  !is.element("shape", theta.names))
                  stop("ci: invalid argument configurations.")
                if (!is.element(x$type, c("GP", "Beta", "Pareto")))
                  stop("ci: invalid argument configurations.")
                sam["scale", ] <- sam["scale", ] - sam["shape",
                  ] * x$threshold
                theta.hat["scale"] <- theta.hat["scale"] - theta.hat["shape"] *
                  x$threshold
                theta.names[theta.names == "scale"] <- "tscale"
                rownames(sam) <- theta.names
                names(theta.hat) <- theta.names
            }
            sam <- sam[which.par, ]
            if (return.samples)
                return(t(sam))
        }
        else if (type == "return.level") {
            pars <- apply(Z, 2, bfun, x = x, y = y, ipar = ipar)[1:np,
                ]
            th.est <- numeric(3)
            if (is.element(x$type, c("PP", "GEV", "Gumbel", "Weibull",
                "Frechet"))) {
                loc <- pars["location", ]
                th.est[1] <- theta.hat["location"]
            }
            else loc <- rep(0, R)
            scale <- pars["scale", ]
            th.est[2] <- theta.hat["scale"]
            if (!is.element(x$type, c("Gumbel", "Exponential"))) {
                shape <- pars["shape", ]
                th.est[3] <- theta.hat["shape"]
            }
            else {
                shape <- rep(1e-10, R)
                th.est[3] <- 1e-08
            }
            if (return.samples)
                out <- t(pars)
            th <- rbind(loc, scale, shape)
            rlfun <- function(theta, p, u, type, npy, rate) rlevd(period = p,
                loc = theta[1], scale = theta[2], shape = theta[3],
                threshold = u, type = type, npy = npy, rate = rate)
            if (x$type == "PP")
                mod <- "GEV"
            else mod <- x$type
            sam <- apply(th, 2, rlfun, p = return.period, u = x$threshold,
                type = mod, npy = x$npy, rate = lam)
            if (is.matrix(sam))
                rownames(sam) <- paste(rownames(sam), "-", x$period.basis,
                  sep = "")
            else sammy.name <- paste(return.period, "-", x$period.basis,
                sep = "")
            if (return.samples) {
                if (is.matrix(sam))
                  out <- cbind(pars, t(sam))
                else {
                  onames <- colnames(out)
                  out <- cbind(out, sam)
                  colnames(out) <- c(onames, sammy.name)
                }
                return(out)
            }
            theta.hat <- rlevd(period = return.period, loc = th.est[1],
                scale = th.est[2], shape = th.est[3], threshold = x$threshold,
                type = x$type, npy = x$npy, rate = lam)
        }
        else stop("ci: invalid type argument.  Must be return.level or parameter.")
        if (is.matrix(sam)) {
            out <- apply(sam, 1, quantile, probs = c(alpha/2,
                1 - alpha/2))
            out.names <- rownames(out)
            out <- rbind(out[1, ], theta.hat, out[2, ])
            rownames(out) <- c(out.names[1], "Estimate", out.names[2])
            colnames(out) <- rownames(sam)
            out <- t(out)
            attr(out, "data.name") <- x$call
            attr(out, "method") <- method.name
            attr(out, "conf.level") <- (1 - alpha) * 100
            attr(out, "R") <- R
            class(out) <- "ci"
            return(out)
        }
        else {
            out <- quantile(sam, probs = c(alpha/2, 1 - alpha/2))
            out <- c(out[1], mean(sam), out[2])
            attr(out, "R") <- R
        }
        if (verbose)
            cat("\n", "Finished fitting model to simulated data.\n")
    }
    else if (method == "proflik") {
        if (x$type == "PP" && !is.null(x$blocks))
            stop("ci: cannot do profile likelihood with blocks.")
        if (tscale)
            stop("ci: invalid argument configurations.")
        if (type == "parameter" && length(which.par) > 1)
            stop("ci: can only do one parameter at a time with profile likelihood method.")
        else if (type == "return.level" && length(return.period) >
            1)
            stop("ci: can only do one return level at a time with profile likelihood method.")
        method.name <- "Profile Likelihood"
        if (verbose) {
            if (x$type != "PP")
                cat("\n", "Calculating profile likelihood.  This may take a few moments.\n")
            else cat("\n", "Calculating profile likelihood.  This may take several moments.\n")
        }
        if (is.null(xrange)) {
            hold2 <- c(ci(x, alpha = alpha, method = "normal",
                type = type, return.period = return.period, which.par = which.par))[c(1,
                3)]
            if (!any(is.na(hold2)))
                xrange <- range(c(hold2, log2(hold2), 4 * hold2,
                  hold2 - 4 * hold2, hold2 + 4 * hold2), finite = TRUE)
            else if (!is.na(hold2[2]))
                xrange <- range(c(p - 2 * abs(log2(abs(p))),
                  hold2[2], 4 * hold2[2], -4 * hold2[2], log2(p)),
                  finite = TRUE)
            else if (!is.na(hold2[1]))
                xrange <- range(c(p - 2 * abs(log2(abs(p))),
                  hold2[1], 4 * hold2[1], -4 * hold2[1], log2(p)),
                  finite = TRUE)
            else if (all(is.na(hold2)))
                xrange <- c(p - 2 * abs(log2(abs(p))), p + 2 *
                  abs(log2(abs(p))))
            if (verbose)
                cat("\n", "Using a range of ", xrange[1], " to ",
                  xrange[2], "\n")
        }
        if (is.null(x$blocks)) {
            if (!is.null(xrange))
                hold <- profliker(x, type = type, xrange = xrange,
                  return.period = return.period, which.par = which.par,
                  nint = nint, plot = verbose, ...)
            else hold <- profliker(x, type = type, return.period = return.period,
                which.par = which.par, nint = nint, plot = verbose,
                ...)
        }
        else stop("Sorry: profile likelihood with blocks is not supported.")
        ma <- -x$results$value
        crit <- ma - 0.5 * qchisq(1 - alpha, 1)
        if (verbose) {
            cat("\n", "Profile likelihood has been calculated.  Now, trying to find where it crosses the critical value = ",
                crit, "\n")
            abline(h = crit, col = "blue")
        }
        crit2 <- ma - 0.5 * qchisq((1 - alpha) + abs(log2(1 -
            alpha))/2, 1)
        print(hold)
        print(crit2)
        id <- hold > crit2
        print(id)
        z <- seq(xrange[1], xrange[2], length = length(hold))
        z <- z[id]
        parlik <- hold[id]
        print("here")
        print(z)
        print(parlik)
        # smth <- spline(z, parlik, n = 200)
        # ind <- smth$y > crit
        # out <- range(smth$x[ind])
        # if (verbose)
        #     abline(v = out, lty = 2, col = "darkblue", lwd = 2)
        # out <- c(out[1], p, out[2])
    }
    # else stop("ci: invalid method argument.")
    # conf.level <- paste(round((1 - alpha) * 100, digits = 2),
    #     "%", sep = "")
    # names(out) <- c(paste(conf.level, " lower CI", sep = ""),
    #     par.name, paste(conf.level, " upper CI", sep = ""))
    # attr(out, "data.name") <- x$call
    # attr(out, "method") <- method.name
    # attr(out, "conf.level") <- (1 - alpha) * 100
    # class(out) <- "ci"
    # return(out)
}
