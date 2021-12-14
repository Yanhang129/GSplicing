#' @export
gsplicing <- function(x, ...) UseMethod("gsplicing")

#' @title Best Subset of Groups Selection in Linear Model
#'
#' @description An implementation of best subset of groups selection (BSGS) in linear model via the splicing approach.
#'
#' @aliases gsplicing
#'
#' @author Yanhang Zhang, Yifan Wang, Xirui Zhao
#'
#' @param x Input matrix, of dimension \eqn{n \times p}; each row is an observation
#' vector and each column is a predictor.
#' @param y The response variable of \code{n} observations.
#' @param group A vector of integers indicating the which group each variable is in.
#' For variables in the same group, they should be located in adjacent columns of \code{x}
#' and their corresponding index in \code{group} should be the same.
#' Denote the first group as \code{1}, the second \code{2}, etc.
#' @param method The method to be used to select the optimal model size. For
#' \code{method = "sequential"}, we solve the BSGS for each model size in \code{T.list}.
#' For \code{method = "gsection"}, we solve the BSGS with model size via a gold-section approach.
#' @param C.max The size of the exchange subset of groups. Default is \code{C.max = 2}.
#' @param tune The type of criterion for choosing the best model size.
#' Available options are \code{"GIC"}, \code{"BIC"}, \code{"AIC"} and \code{"cv"}.
#' Default is \code{tune = "GIC"}.
#' @param T.list An integer vector representing the alternative model sizes.
#' Only used for \code{method = "sequential"}. Default is \code{T.list = 1:min(N, round(n/(log(p)*min(table(group))))))}.
#' @param T.min The minimum value of model sizes. Only used for \code{method = "gsection"}. Default is \code{T.min = 1}.
#' @param T.max The maximum value of model sizes. Only used for \code{method = "gsection"}. Default is \code{T.max = min(N, round(n/(log(p)*min(table(group)))))}.
#' @param pi Threshold of the difference of the loss function in each iteration.
#' @param weight Observation weights. Default is \code{weight = 1} for each observation as default.
#' @param max.iter  The maximum number of performing splicing algorithm.
#' In most of the case, only a few times of splicing iteration can guarantee the convergence.
#' Default is \code{max.iter = 10}.
#' @param nfolds The number of folds in cross-validation. Default is \code{nfolds = 5}.
#' @param ... further arguments to be passed to or from methods.
#' 
#' @return A S3 \code{gsplicing} class object, which is a \code{list} with the following components:
#' \item{beta}{A \eqn{p}-by-\code{length(T.list)} matrix of coefficients, stored in column format.}
#' \item{intercept}{An intercept vector of \code{length(T.list)}.
#' \item{A.out}{The selected groups given model size in \code{T.list}.}
#' \item{best_group}{The selected groups with model size equal to \code{best_model_size}.}
#' \item{best_model_size}{The model size attaining the smallest \code{ic}.}
#' \item{loss}{The values of the loss function for each model size in \code{T.list}.}
#' \item{ic}{The values of the specified criterion for each model size in \code{T.list}.}
#' \item{T.list}{The actual \code{T.list} values used.}
#' \item{method}{The method used to select the best model size.}
#' \item{ic_type}{The type of criterion for choosing the best model size.}
#' 
#' @details
#' Best subset of groups selection aims to  identify
#' a small subset of groups to achieve the best interpretability on the
#' response variable. We call this problem as Best Subset of Groups
#' Selection (BSGS), BSGS problem under the support size \eqn{T} is:
#' \deqn{min_{\beta}\frac{1}{2n}||y-X\beta||^2_2 s.t.||\beta||_{0,2}<= T,}
#' see Zhang(2021) for details. To find the optimal support size \eqn{T},
#' we provide various criterion like GIC, AIC, BIC and cross-validation to determine it.
#'
#' @references Certifiably Polynomial Algorithm for Best Group Subset Selection. Zhang, Yanhang, Junxian Zhu, Jin Zhu, and Xueqin Wang (2021). arXiv preprint arXiv:2104.12576.
#' @references A polynomial algorithm for best-subset selection problem. Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, Xueqin Wang. Proceedings of the National Academy of Sciences Dec 2020, 117 (52) 33117-33123; DOI: 10.1073/pnas.2014241117
#'
#' @export
#' @rdname gsplicing
#' @method gsplicing default
#' @examples
#' \donttest{
#' library(GSplicing)
#' n <- 200
#' J <- 100
#' k <- 5
#' model.size <- 5
#' data <- generate.group(n, J, k, model.size)
#' fit <- gsplicing(data$x, data$y, data$group)
#' all(fit$best_group == data$true.group)
#' }
gsplicing.default <- function(x, y, group,  
                      method = c("sequential", "gsection"), C.max = 2,
                      tune = c("GIC", "BIC", "AIC", "cv"),
                      T.list, T.min = 1, T.max, pi,
                      weight = rep(1, nrow(x)),
                      max.iter = 10, nfolds = 5, ...)
{
  tune = match.arg(tune)
  method = match.arg(method)
  weight = weight/mean(weight)
  if(missing(group)) group <- 1:ncol(x)
  p <- ncol(x)
  n <- nrow(x)
  N <- length(unique(group))
  if (length(group)!= ncol(x)) stop("The length of group should be the same with ncol(x)")
  pmin <- min(table(group))
  if(missing(T.list)) 
  {
    T.list <- 1:min(N, round(n/(log(p)*pmin)))
  } else {
    T.list <- sort(T.list)
  }
  if (missing(T.max)) T.max <- min(N, round(n/(log(p)*pmin)))
  if (missing(pi)) pi <- 0.1*(log(p)*log(log(n)))/n
  tune <- match.arg(tune)
  ic_type <- switch(tune,
                    "AIC" = 1,
                    "BIC" = 2,
                    "GIC" = 3,
                    "BGIC" = 4,
                    "cv" = 5)
  is_cv <- ifelse(tune == "cv", TRUE, FALSE)
  if(method == "sequential")
  { 
    path_type <- 1
  } else {
    path_type <- 2
  }
  
  if(!is.matrix(x)) x <- as.matrix(x)
  vn <- colnames(x)
  orderGi <- order(group)
  x <- x[, orderGi]
  vn <- vn[orderGi]
  group <- group[orderGi]
  gi <- unique(group)
  index <- match(gi, group)-1

  res <- gsplicingCpp(x, y, weight = weight, max_iter = max.iter, exchange_num = C.max, path_type = path_type,
                 ic_type = ic_type, is_cv = is_cv, K = nfolds, sequence = T.list,
                 s_min = T.min, s_max = T.max, g_index = index, tau = pi)
  
  if(is_cv == TRUE) {
    names(res)[which(names(res) == "ic")] <- "cv"
  }

  if (path_type == 1)
  { 
    res$T.list <- T.list
  } else {
    res$T.max <- T.max
    res$T.min <- T.min
  }
  res$method <- method
  res$ic_type <- ifelse(is_cv == TRUE, "cv", c("AIC", "BIC", "GIC", "BGIC")[ic_type])
  class(res) <- 'GSplicing'
  return(res)
}