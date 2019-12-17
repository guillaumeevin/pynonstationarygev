
# todo: report bug on exTremes, fix is around line 510 where I set theta[4] = 0 (and it was Nan before)
# it was important to do that to make the code work in the Gumbel case
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
        return(ci.rl.ns.fevd.mle_fixed(x = x, alpha = alpha, return.period = return.period,
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
                shape <- rep(0, R)
                th.est[3] <- 0
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
        id <- hold > crit2
        z <- seq(xrange[1], xrange[2], length = length(hold))
        z <- z[id]
        parlik <- hold[id]
        smth <- spline(z, parlik, n = 200)
        ind <- smth$y > crit
        out <- range(smth$x[ind])
        if (verbose)
            abline(v = out, lty = 2, col = "darkblue", lwd = 2)
        out <- c(out[1], p, out[2])
    }
    else stop("ci: invalid method argument.")
    conf.level <- paste(round((1 - alpha) * 100, digits = 2),
        "%", sep = "")
    names(out) <- c(paste(conf.level, " lower CI", sep = ""),
        par.name, paste(conf.level, " upper CI", sep = ""))
    attr(out, "data.name") <- x$call
    attr(out, "method") <- method.name
    attr(out, "conf.level") <- (1 - alpha) * 100
    class(out) <- "ci"
    return(out)
# }}
}


ci.rl.ns.fevd.mle_fixed <- function (x, alpha = 0.05, return.period = 100, method = c("normal"),
    verbose = FALSE, qcov = NULL, qcov.base = NULL, ...)
{
    method <- tolower(method)
    method <- match.arg(method)
    par.name <- paste(return.period, "-", x$period.basis, " return level",
        sep = "")
    if (verbose) {
        cat("\n", "Preparing to calculate ", (1 - alpha) * 100,
            "% CI for ", paste(return.period, "-", x$period.basis,
                " return level", sep = ""), "\n")
        cat("\n", "Model is non-stationary.\n")
        cat("\n", "Using Normal Approximation Method.\n")
    }
    if (method == "normal")
        method.name <- "Normal Approx."
    if (method == "normal") {
        res <- return.level.ns.fevd.mle_fixed(x = x, return.period = return.period,
            ..., do.ci = FALSE, verbose = verbose, qcov = qcov,
            qcov.base = qcov.base)
        z.alpha <- qnorm(alpha/2, lower.tail = FALSE)
        cov.theta <- parcov.fevd(x)
        if (is.null(cov.theta))
            stop("ci: Sorry, unable to calculate the parameter covariance matrix.  Maybe try a different method.")
        var.theta <- diag(cov.theta)
        if (any(var.theta < 0))
            stop("ci: negative Std. Err. estimates obtained.  Not trusting any of them.")
        grads <- t(rlgrad.fevd(x, period = return.period, qcov = qcov,
            qcov.base = qcov.base))
        se.theta <- sqrt(diag(t(grads) %*% cov.theta %*% grads))
        out <- cbind(c(res) - z.alpha * se.theta, c(res), c(res) +
            z.alpha * se.theta, se.theta)
        if (length(return.period) > 1)
            rownames(out) <- par.name
        else rownames(out) <- NULL
        conf.level <- paste(round((1 - alpha) * 100, digits = 2),
            "%", sep = "")
        colnames(out) <- c(paste(conf.level, " lower CI", sep = ""),
            "Estimate", paste(conf.level, " upper CI", sep = ""),
            "Standard Error")
    }
    else if (method == "boot") {
        stop("ci.rl.ns.fevd.mle: Sorry, this functionality has not yet been added.")
    }
    attr(out, "data.name") <- x$call
    attr(out, "method") <- method.name
    attr(out, "conf.level") <- (1 - alpha) * 100
    class(out) <- "ci"
    return(out)
}

return.level.ns.fevd.mle_fixed <- function (x, return.period = c(2, 20, 100), ..., alpha = 0.05,
    method = c("normal"), do.ci = FALSE, verbose = FALSE, qcov = NULL,
    qcov.base = NULL)
{
    if (missing(return.period))
        return.period <- 100
    if (do.ci && method != "normal")
        stop("return.level.ns.fevd.mle: only normal approximation CI calculations currently available for nonstationary return levels.")
    model <- x$type
    if (!(tolower(model) %in% c("pp", "gev", "weibull", "frechet",
        "gumbel")))
        stop("return.level.ns.fevd.mle: not implemented for GP, beta, pareto, exponential models.")
    if (do.ci && length(return.period) > 1 && nrow(qcov) > 1)
        stop("return.level.ns.fevd.mle:: Cannot calculate confidence intervals for multiple return periods and multiple covariate values simultaneously.")
    if (!do.ci) {
        if (model == "PP") {
            mod2 <- "GEV"
        }
        else mod2 <- model
        p <- x$results$par
        pnames <- names(p)
        if (is.fixedfevd(x))
            stop("return.level.ns.fevd.mle: this function is for nonstationary models.")
        if (is.null(qcov))
            stop("return.level.ns.fevd.mle: qcov required for this function.")
        if (!is.matrix(qcov))
            qcov <- matrix(qcov, nrow = 1)
        if (!is.qcov(qcov))
            qcov <- make.qcov(x = x, vals = qcov, nr = nrow(qcov))
        if (!is.null(qcov.base)) {
            if (!is.matrix(qcov.base))
                qcov.base <- matrix(qcov.base, nrow = 1)
            if (nrow(qcov) != nrow(qcov.base) || ncol(qcov) !=
                ncol(qcov.base))
                stop("return.level.ns.fevd.mle: qcov and qcov.base must have the same number of covariates and values.")
            if (!is.qcov(qcov.base))
                qcov.base <- make.qcov(x = x, vals = qcov.base,
                  nr = nrow(qcov.base))
        }
        loc.id <- 1:x$results$num.pars$location
        sc.id <- (1 + x$results$num.pars$location):(x$results$num.pars$location +
            x$results$num.pars$scale)
        sh.id <- (1 + x$results$num.pars$location + x$results$num.pars$scale):(x$results$num.pars$location +
            x$results$num.pars$scale + x$results$num.pars$shape)
        loc <- qcov[, loc.id, drop = FALSE] %*% p[loc.id]
        sc <- qcov[, sc.id, drop = FALSE] %*% p[sc.id]
        sh <- qcov[, sh.id, drop = FALSE] %*% p[sh.id]
        if (!is.null(qcov.base)) {
            loc.base <- qcov.base[, loc.id, drop = FALSE] %*%
                p[loc.id]
            sc.base <- qcov.base[, sc.id, drop = FALSE] %*% p[sc.id]
            sh.base <- qcov.base[, sh.id, drop = FALSE] %*% p[sh.id]
        }
        if (x$par.models$log.scale) {
            sc <- exp(sc)
            if (!is.null(qcov.base))
                sc.base <- exp(sc.base)
        }
        theta <- cbind(qcov[, "threshold"], loc, sc, sh)
        rlfun2 <- function(th, pd, type, npy, rate) rlevd_fixed(pd,
            loc = th[2], scale = th[3], shape = th[4], threshold = th[1],
            type = type, npy = npy)
        theta[4] = 0
        res <- apply(theta, 1, rlfun2, pd = return.period, type = mod2,
            npy = x$npy)
        if (!is.null(qcov.base)) {
            theta.base <- cbind(qcov.base[, "threshold"], loc.base,
                sc.base, sh.base)
            res <- res - apply(theta.base, 1, rlfun2, pd = return.period,
                type = mod2, npy = x$npy)
        }
        res <- t(matrix(res, nrow = length(return.period)))
        colnames(res) <- paste(return.period, "-", x$period.basis,
            " level", sep = "")
        attr(res, "return.period") <- return.period
        attr(res, "data.name") <- x$data.name
        attr(res, "fit.call") <- x$call
        attr(res, "call") <- match.call()
        attr(res, "fit.type") <- x$type
        attr(res, "data.assumption") <- "non-stationary"
        attr(res, "period") <- x$period.basis
        attr(res, "units") <- x$units
        attr(res, "qcov") <- deparse(substitute(qcov))
        attr(res, "class") <- "return.level"
        if (!is.null(qcov.base)) {
            attr(res, "qcov.base") <- deparse(substitute(qcov.base))
            attr(res, "class") <- "return.level.diff"
        }
        return(res)
    }
    else {
        out <- ci.rl.ns.fevd.mle(x = x, alpha = alpha, return.period = return.period,
            qcov = qcov, qcov.base = qcov.base, ...)
        return(out)
    }
}

rlevd_fixed <- function (period, loc = 0, scale = 1, shape = 0, threshold = 0,
    type = c("GEV", "GP", "PP", "Gumbel", "Frechet", "Weibull",
        "Exponential", "Beta", "Pareto"), npy = 365.25, rate = 0.01)
{
    if (any(period <= 1))
        stop("rlevd: invalid period argument.  Must be greater than 1.")
    type <- match.arg(type)
    type <- tolower(type)
    if (missing(loc))
        loc <- 0
    else if (is.null(loc))
        loc <- 0
    if (is.element(type, c("gumbel", "weibull", "frechet"))) {
        if (type == "gumbel" && shape != 0) {
            warning("rlevd: shape is not zero, but type is Gumbel.  Re-setting shape parameter to zero.")
            shape <- 0
            type <- "gev"
        }
        else if (type == "gumbel")
            type <- "gev"
        else if (type == "frechet" && shape <= 0) {
            if (shape == 0) {
                warning("rlevd: shape is zero, but type is Frechet!  Re-setting type to Gumbel.")
                shape <- 0
            }
            else {
                warning("rlevd: type is Frechet, but shape < 0.  Negating shape to force it to be Frechet.")
                shape <- -shape
            }
            type <- "gev"
        }
        else if (type == "frechet")
            type <- "gev"
        else if (type == "weibull" && shape >= 0) {
            if (shape == 0) {
                warning("rlevd: shape is zero, but type is Weibull!  Re-setting type to Gumbel.")
                shape <- 0
            }
            else {
                warning("rlevd: type is Weibull, but shape > 0.  Negating shape to force it to be Weibull.")
                shape <- -shape
            }
            type <- "gev"
        }
        else if (type == "weibull")
            type <- "gev"
    }
    if (is.element(type, c("beta", "pareto", "exponential"))) {
        if (type == "exponential" && shape != 0) {
            warning("rlevd: shape is not zero, but type is Exponential.  Re-setting shape parameter to zero.")
            shape <- 0
            type <- "gp"
        }
        else if (type == "exponential")
            type <- "gp"
        else if (type == "beta" && shape >= 0) {
            if (shape == 0) {
                warning("rlevd: shape is zero, but type is Beta!  Re-setting type to Exponential.")
                shape <- 0
            }
            else {
                warning("rlevd: type is Beta, but shape > 0.  Negating shape to force it to be Beta.")
                shape <- -shape
            }
            type <- "gp"
        }
        else if (type == "beta")
            type <- "gp"
        else if (type == "pareto" && shape <= 0) {
            if (shape == 0) {
                warning("rlevd: shape is zero, but type is Pareto!  Re-setting type to Exponential.")
                shape <- 0
            }
            else {
                warning("rlevd: type is Pareto, but shape < 0.  Negating shape to force it to be Pareto.")
                shape <- -shape
            }
            type <- "gp"
        }
        else if (type == "pareto")
            type <- "gp"
    }
    if (is.element(type, c("gev", "pp"))) {
        p <- 1 - 1/period
        res <- qevd(p = p, loc = loc, scale = scale, shape = shape,
            type = "GEV")
    }
    else if (type == "gp") {
        m <- period * npy * rate
        if (shape == 0)
            res <- threshold + scale * log(m)
        else res <- threshold + (scale/shape) * (m^shape - 1)
    }
    names(res) <- as.character(period)
    return(res)
}
