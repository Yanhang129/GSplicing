# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

gsplicingCpp <- function(x, y, weight, max_iter, exchange_num, path_type, ic_type, is_cv, K, sequence, s_min, s_max, g_index, tau) {
    .Call('_GSplicing_gsplicingCpp', PACKAGE = 'GSplicing', x, y, weight, max_iter, exchange_num, path_type, ic_type, is_cv, K, sequence, s_min, s_max, g_index, tau)
}

Orthonormalize <- function(X, y, index, gsize, n, p, N, weights, meanx, meany) {
    .Call('_GSplicing_Orthonormalize', PACKAGE = 'GSplicing', X, y, index, gsize, n, p, N, weights, meanx, meany)
}

