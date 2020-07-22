# Title     : TODO
# Objective : TODO
# Created by: erwan
# Created on: 15/05/2020
##
## Exported symobls in package `gnFit`
##

## Exported package methods

gnfit_fixed <- function (dat, dist, df = NULL, pr = NULL, threshold = NULL)
{
    dat <- as.numeric(dat)
    x <- NULL
    z <- list()
    op <- par(mfrow = c(1, 2))
    if (is.null(pr)) {
        if (dist == "gev" | dist == "gpd")
            stop("Enter Parameters!")
        else if (dist == "t") {
            if (df > 2) {
                loc <- mean(dat)
                sc <- sqrt((df - 2) * var(dat)/df)
                xdat <- (dat - loc)/sc
                prob <- pt(xdat, df)
            }
            else stop("DF must be > 2")
        }
        else if (dist == "gum") {
            sc <- sqrt(6 * var(dat)/pi^2)
            loc <- mean(dat) - 0.577 * sc
            pr <- c(loc, sc, 0)
            prob <- gevf(pr, dat)
        }
        else {
            loc <- mean(dat)
            ifelse(dist == "norm", sc <- sd(dat), NA)
            ifelse(dist == "laplace", sc <- sqrt(var(dat)/2),
                NA)
            ifelse(dist == "logis", sc <- sqrt(3 * var(dat)/pi^2),
                NA)
            prob <- get(paste0("p", dist))(dat, loc, sc)
        }
    }
    else {
        if (dist == "gev") {
            prob <- gevf(pr, dat)
        }
        else if (dist == "gum") {
            pr[3] <- 0
            prob <- gevf(pr, dat)
        }
        else if (dist == "gpd")
            if (!is.null(threshold)) {
                u <- threshold
                dat <- dat[dat > u]
                prob <- gpdf(pr, u, dat)
            }
            else stop("threshold is missing!")
    }
    n <- length(dat)
    k <- seq(1:n)
    qnor <- qnorm(sort(prob))
    pnor <- pnorm((qnor - mean(qnor))/sd(qnor))
    w <- round((sum((pnor - (2 * k - 1)/(2 * n))^2) + 1/(12 *
        n)) * (1 + 0.5/n), 4)
    if (w < 0.0275) {
        pval <- 1 - exp(-13.953 + 775.5 * w - 12542.61 * w^2)
    }
    else if (w < 0.051) {
        pval <- 1 - exp(-5.903 + 179.546 * w - 1515.29 * w^2)
    }
    else if (w < 0.092) {
        pval <- exp(0.886 - 31.62 * w + 10.897 * w^2)
    }
    else if (w < 1.1) {
        pval <- exp(1.111 - 34.242 * w + 12.832 * w^2)
    }
    else {
        pval <- 0 # I added that to avoid the code to crash
        warning("p-value is smaller than 7.37e-10")
    }
    z$Wpval <- pval
    A <- (-n - sum((2 * k - 1) * log(pnor) + (2 * n + 1 - 2 *
        k) * log(1 - pnor))/n) * (1 + 0.75/n + 2.25/n^2)
    A <- round((1 + 0.75/n + 2.25/n^2) * A, 4)
    if (A < 0.2) {
        pval <- 1 - exp(-13.436 + 101.14 * A - 223.73 * A^2)
    }
    else if (A < 0.34) {
        pval <- 1 - exp(-8.318 + 42.796 * A - 59.938 * A^2)
    }
    else if (A < 0.6) {
        pval <- exp(0.9177 - 4.279 * A - 1.38 * A^2)
    }
    else if (A < 10) {
        pval <- exp(1.2937 - 5.709 * A + 0.0186 * A^2)
    }
    else {
        pval <- 0 # I added that to avoid the code to crash
        warning("p-value is smaller than 7.37e-10")
    }
    z$Apval <- pval
    z$Cram <- w
    z$Ander <- A
    invisible(z)
}




## Package Data

# none


## Package Info

.skeleton_package_title = "Goodness of Fit Test for Continuous Distribution Functions"

.skeleton_package_version = "0.2.0"

.skeleton_package_depends = ""

.skeleton_package_imports = "ismev,rmutil"


## Internal

.skeleton_version = 5


## EOF
