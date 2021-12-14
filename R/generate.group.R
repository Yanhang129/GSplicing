#' @title Generate simulated data
#'
#' @description Generate simulated data for group linear model.
#'
#' @param n The number of observations.
#' @param J The number of groups of interest.
#' @param k The group size of each group. Only even group structure is allowed here.
#' @param model.size The number of important groups in the underlying regression model. 
#' @param rho A parameter used to characterize the pairwise correlation in
#' predictors. Default is \code{0.5}..
#' @param cortype The correlation structure.
#' \code{cortype = 1} denotes the independence structure,
#' where the covariance matrix has \eqn{(i,j)} entry equals \eqn{I(i \neq j)}.
#' \code{cortype = 2} denotes the exponential structure,
#' where the covariance matrix has \eqn{(i,j)} entry equals \eqn{rho^{|i-j|}}.
#' code{cortype = 3} denotes the constant structure,
#' where the non-diagnoal entries of covariance
#' matrix are \eqn{rho} and diagonal entries are 1.
#' @param sigma1 The value controlling the strength of the gaussian noise. A large value implies strong noise. Default \code{sigma1 = 1}.
#' @param sigma1 The value controlling the strength of the coefficients. A large value implies large coefficients. Default \code{sigma2 = 1}.
#' @param seed random seed. Default: \code{seed = 1}.
#' 
#' @return A \code{list} object comprising:
#' \item{x}{Design matrix of predictors.}
#' \item{y}{Response variable.}
#' \item{beta}{The coefficients used in the underlying regression model.}
#' \item{group}{The group index of each variable.}
#' \item{true.group}{The important groups in the group linear model.}
#'
#' @author Yanhang Zhang
#'
#' @export
#'
#' @examples
#'
#' # Generate simulated data
#' n <- 200
#' J <- 100
#' k <- 5
#' model.size <- 5
#' data <- generate.group(n, J, k, model.size)
#' str(data)

generate.group <- function(n,
                           J,
                           k,
                           model.size = 5,
                           cortype = 1,
                           rho = 0.5,
                           sigma1 = 1,
                           sigma2 = 1,
                           seed = 1) {
  set.seed(seed)
  group <- rep(1:J, each = k)
  
  if (cortype == 1) {
    Sigma <- diag(J)
  } else if (cortype == 2) {
    Sigma <- matrix(0, J, J)
    Sigma <- rho ^ (abs(row(Sigma) - col(Sigma)))
  } else if (cortype == 3) {
    Sigma <- matrix(rho, J, J)
    diag(Sigma) <- 1
  }
  if (cortype == 1) {
    x <- matrix(rnorm(n * J), nrow = n, ncol = J)
  } else {
    x <- MASS::mvrnorm(n, rep(0, J), Sigma)
  }
  z <- matrix(rnorm(n * J * k), nrow = n)
  z <- sapply(1:(J * k), function(i) {
    g <- floor((i - 1) / k) + 1
    return((x[, g] + z[, i]) / sqrt(2))
  })
  x <- matrix(unlist(z), n)
  beta <- rep(0, J * k)
  true.group <- sort(sample(1:J, model.size))
  nonzero <- as.vector(sapply(true.group, function(i) {
    return(((i - 1) * k + 1):(i * k))
  }))
  coef <- rep(0, model.size * k)
  for (i in 1:model.size) {
    temp <- stats::rnorm(k + 1, 0, sigma2)
    coef[((i - 1) * k + 1):(i * k)] <- (temp - mean(temp))[-1]
  }
  beta[nonzero] <- coef
  y <- x %*% beta + rnorm(n, 0, sigma1)
  set.seed(NULL)
  colnames(x) <- paste0("x", 1:(J * k))
  return(list(
    x = x,
    y = y,
    beta = beta,
    group = group,
    true.group = true.group
  ))
}