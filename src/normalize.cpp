//
// Created by jk on 2020/3/8.
//
//#define R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#include <algorithm>
#include <vector>
#include <iostream>
using namespace Rcpp;
using namespace std;

// [[Rcpp::export]]
std::vector<Eigen::MatrixXd> Orthonormalize(Eigen::MatrixXd& X, Eigen::VectorXd& y, Eigen::VectorXi& index, Eigen::VectorXi& gsize, int n, int p, int N, Eigen::VectorXd& weights, Eigen::VectorXd& meanx, double& meany){
  int size;
  for (int i=0;i<n;i++) {
    X.row(i) = X.row(i)*sqrt(weights(i));
  }
  meanx = X.colwise().mean();
  X = X.rowwise()-meanx.transpose();
  std::vector<Eigen::MatrixXd> mat(N);
  for (int i=0;i<N;i++) {
    size = gsize(i);
    if (size == 1) {
      Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(1, 1);
      temp(0, 0) = X.col(index(i)).norm()/sqrt(n);
      mat[i] = temp;
      X.col(index(i)) = X.col(index(i))/temp(0, 0);
    }
    else
    {
      Eigen::MatrixXd X_ind = X.block(0, index(i), n, size);
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(X_ind.transpose()*X_ind/n, Eigen::ComputeThinU | Eigen::ComputeThinV);
      Eigen::MatrixXd Sigma = Eigen::MatrixXd::Zero(size, size);
      Sigma.diagonal() =  svd.singularValues();
      Sigma = Sigma.ldlt().solve(Eigen::MatrixXd::Identity(size, size)).sqrt();
      Eigen::MatrixXd U = svd.matrixU();
      Eigen::MatrixXd temp = U*Sigma;
      X.block(0, index(i), n, size) = X_ind*temp;
      mat[i] = temp;
    }
  }
  y = y.cwiseProduct(weights);
  meany = y.mean();
  y = y.array() - meany;
  return mat;
}
