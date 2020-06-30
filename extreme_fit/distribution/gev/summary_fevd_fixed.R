# Title     : TODO
# Objective : TODO
# Created by: erwan
# Created on: 30/06/2020
summary.fevd.mle_fixed <- function (object, ...)
{
    x <- object
    a <- list(...)
    out <- list()
    cov.theta <- se.theta <- NULL
    if (!is.null(a$silent))
        silent <- a$silent
    else silent <- FALSE
    if (!silent) {
        cat("\n")
        print(x$call)
        cat("\n")
        print(paste("Estimation Method used: ", x$method, sep = ""))
        cat("\n")
    }
    if (!silent)
        cat("\n", "Negative Log-Likelihood Value: ", x$results$value,
            "\n\n")
    theta.hat <- x$results$par
    theta.names <- names(theta.hat)
    if (is.element("log.scale", theta.names)) {
        theta.hat[theta.names == "log.scale"] <- exp(theta.hat[theta.names ==
            "log.scale"])
        theta.names[theta.names == "log.scale"] <- "scale"
        names(theta.hat) <- theta.names
        phiU <- FALSE
    }
    else phiU <- x$par.models$log.scale
    out$par <- theta.hat
    np <- length(theta.hat)
    designs <- setup.design(x)
    if (!is.null(x$data.pointer))
        xdat <- get(x$data.pointer)
    else xdat <- x$x
    # cov.theta <- parcov.fevd(x)
    # out$cov.theta <- cov.theta
    # if (!is.null(cov.theta)) {
    #     se.theta <- sqrt(diag(cov.theta))
    #     names(se.theta) <- theta.names
    #     out$se.theta <- se.theta
    # }
    if (!silent) {
        cat("\n", "Estimated parameters:\n")
        print(theta.hat)
    }
    if (!is.null(se.theta)) {
        if (!silent) {
            cat("\n", "Standard Error Estimates:\n")
            print(se.theta)
        }
        theta.hat <- rbind(theta.hat, se.theta)
        if (is.matrix(theta.hat) && dim(theta.hat)[1] == 2)
            rownames(theta.hat) <- c("Estimate", "Std. Error")
        if (is.matrix(theta.hat))
            colnames(theta.hat) <- theta.names
        else names(theta.hat) <- theta.names
        if (!silent && !is.null(cov.theta)) {
            cat("\n", "Estimated parameter covariance matrix.\n")
            print(cov.theta)
        }
    }
    nllh <- x$results$value
    out$nllh <- nllh
    if (is.element(x$type, c("GEV", "Gumbel", "Weibull", "Frechet")))
        n <- x$n
    else {
        y <- c(datagrabber(x, cov.data = FALSE))
        n <- sum(y > x$threshold)
    }
    out$AIC <- 2 * nllh + 2 * np
    out$BIC <- 2 * nllh + np * log(n)
    if (!silent) {
        cat("\n", "AIC =", out$AIC, "\n")
        cat("\n", "BIC =", out$BIC, "\n")
    }
    invisible(out)
}

