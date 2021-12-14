#include <Rcpp.h>
#include <RcppEigen.h>
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppEigen)]]
#include <iostream>
#include "Data.h"
#include "Algorithm.h"
#include "Metric.h"
#include "path.h"
#include "utilities.h"
#include <vector>
using namespace Rcpp;
using namespace std;

// [[Rcpp::export]]
List gsplicingCpp(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd weight,
             int max_iter, int exchange_num,
             int path_type, int ic_type, bool is_cv, int K,
             Eigen::VectorXi sequence, int s_min, int s_max,
             Eigen::VectorXi g_index, double tau) {
  srand(123);
  Data data(x, y, weight, g_index);
  double coef_init = 0.0;
  data.add_weight();
  Algorithm *algorithm = new GSplicingLm(data, max_iter, exchange_num);
  algorithm->set_warm_start(1);

  Metric *metric = new LmMetric(ic_type, is_cv, K);
  
  if (is_cv) {
    metric->set_cv_train_test_mask(data.get_n());
  }
  
  List result;
  if (path_type == 1) 
  {
    result = sequential_path(data, algorithm, metric, sequence, coef_init, tau);
  }
  else
  {
    result = gs_path(data, algorithm, metric, s_min, s_max, coef_init, tau);
  }
  algorithm -> ~Algorithm();
  metric -> ~Metric();
  return result;
}


