// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// gsplicingCpp
List gsplicingCpp(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd weight, int max_iter, int exchange_num, int path_type, int ic_type, bool is_cv, int K, Eigen::VectorXi sequence, int s_min, int s_max, Eigen::VectorXi g_index, double tau);
RcppExport SEXP _GSplicing_gsplicingCpp(SEXP xSEXP, SEXP ySEXP, SEXP weightSEXP, SEXP max_iterSEXP, SEXP exchange_numSEXP, SEXP path_typeSEXP, SEXP ic_typeSEXP, SEXP is_cvSEXP, SEXP KSEXP, SEXP sequenceSEXP, SEXP s_minSEXP, SEXP s_maxSEXP, SEXP g_indexSEXP, SEXP tauSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type x(xSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type y(ySEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type weight(weightSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< int >::type exchange_num(exchange_numSEXP);
    Rcpp::traits::input_parameter< int >::type path_type(path_typeSEXP);
    Rcpp::traits::input_parameter< int >::type ic_type(ic_typeSEXP);
    Rcpp::traits::input_parameter< bool >::type is_cv(is_cvSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type sequence(sequenceSEXP);
    Rcpp::traits::input_parameter< int >::type s_min(s_minSEXP);
    Rcpp::traits::input_parameter< int >::type s_max(s_maxSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type g_index(g_indexSEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    rcpp_result_gen = Rcpp::wrap(gsplicingCpp(x, y, weight, max_iter, exchange_num, path_type, ic_type, is_cv, K, sequence, s_min, s_max, g_index, tau));
    return rcpp_result_gen;
END_RCPP
}
// Orthonormalize
std::vector<Eigen::MatrixXd> Orthonormalize(Eigen::MatrixXd& X, Eigen::VectorXd& y, Eigen::VectorXi& index, Eigen::VectorXi& gsize, int n, int p, int N, Eigen::VectorXd& weights, Eigen::VectorXd& meanx, double& meany);
RcppExport SEXP _GSplicing_Orthonormalize(SEXP XSEXP, SEXP ySEXP, SEXP indexSEXP, SEXP gsizeSEXP, SEXP nSEXP, SEXP pSEXP, SEXP NSEXP, SEXP weightsSEXP, SEXP meanxSEXP, SEXP meanySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type y(ySEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi& >::type index(indexSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi& >::type gsize(gsizeSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type weights(weightsSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type meanx(meanxSEXP);
    Rcpp::traits::input_parameter< double& >::type meany(meanySEXP);
    rcpp_result_gen = Rcpp::wrap(Orthonormalize(X, y, index, gsize, n, p, N, weights, meanx, meany));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_GSplicing_gsplicingCpp", (DL_FUNC) &_GSplicing_gsplicingCpp, 14},
    {"_GSplicing_Orthonormalize", (DL_FUNC) &_GSplicing_Orthonormalize, 10},
    {NULL, NULL, 0}
};

RcppExport void R_init_GSplicing(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
