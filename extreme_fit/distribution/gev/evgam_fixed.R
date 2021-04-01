# Title     : TODO
# Objective : TODO
# Created by: erwan
# Created on: 01/04/2021
library(evgam)
attach(loadNamespace("evgam"), name = "evgam_all")
evgam_fixed <- function(formula, data, family = "gev", correctV = TRUE, rho0 = 0, inits = NULL, outer = "bfgs", control = NULL, removeData = FALSE, trace = 0, knots = NULL, maxdata = 1e+20, maxspline = 1e+20, compact = FALSE, ald.args = list(), exi.args = list(), pp.args = list(), sandwich.args = list()) {
  family.info <- .setup.family(family, pp.args)
  if (is.null(family.info$lik.fns$d340)) outer <- "fd"
  formula <- .setup.formulae(formula, family.info$npar, family.info$npar2, data, trace)
  response.name <- attr(formula, "response.name")
  temp.data <- .setup.data(data, response.name, formula, family, family.info$nms, removeData, exi.args, ald.args, pp.args, knots, maxdata, maxspline, compact, sandwich.args, tolower(outer), trace)
  data <- temp.data$data
  beta <- .setup.inner.inits(inits, temp.data$lik.data, family.info$lik.fns, family.info$npar, family)
  lik.data <- .sandwich(temp.data$lik.data, beta)
  if (trace > 0 & lik.data$adjust > 0) cat(paste("\n Sandwich correct lambda =", signif(lik.data$k, 3), "\n"))
  smooths <- length(temp.data$gotsmooth) > 0
  if (smooths) {
    S.data <- .joinSmooth(temp.data$gams)
    nsp <- length(attr(S.data, "Sl"))
    if (is.null(rho0)) {
      diagSl <- sapply(attr(S.data, "Sl"), diag)
      rho0 <- apply(diagSl, 2, function(y) uniroot(.guess, c(-100, 100), d = attr(beta, "diagH"), s = y)$root)
    } else {
      if (length(rho0) == 1) rho0 <- rep(rho0, nsp)
    }
    lik.data$S <- .makeS(S.data, exp(rho0))
    fit.reml <- .outer(rho0, beta, family.info$lik.fns, lik.data, S.data, control, correctV, lik.data$outer, trace)
    sp <- exp(fit.reml$par)
    lik.data$S <- .makeS(S.data, sp)
  } else {
    S.data <- NULL
    fit.reml <- .outer.nosmooth(beta, family.info$lik.fns, lik.data, control, trace)
  }
  VpVc <- .VpVc(fit.reml, family.info$lik.fns, lik.data, S.data, correctV = correctV, sandwich = temp.data$sandwich, smooths = smooths, trace = trace)
  edf <- .edf(fit.reml$beta, family.info$lik.fns, lik.data, VpVc, temp.data$sandwich)
  names(temp.data$gams) <- family.info$nms
  gams <- .swap(fit.reml, temp.data$gams, lik.data, VpVc, temp.data$gotsmooth, edf, smooths)
  gams <- .finalise(gams, data, family.info$lik.fns, lik.data, S.data, fit.reml, VpVc, family, temp.data$gotsmooth, formula, response.name, removeData, edf)
  return(gams)
}

