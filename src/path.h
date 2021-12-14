//
// Created by jtwok on 2020/3/8.
//
#ifndef SRC_PATH_H
#define SRC_PATH_H

#include <RcppEigen.h>
#include <Rcpp.h>
#include "Data.h"
#include "Algorithm.h"
#include "Metric.h"
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;

List sequential_path(Data &data, Algorithm *algorithm, Metric *metric, Eigen::VectorXi sequence, double coef_init, double tau);

List gs_path(Data &data, Algorithm *algorithm, Metric *metric,
             int s_min, int s_max, double coef_init, double tau);



#endif //SRC_PATH_H
