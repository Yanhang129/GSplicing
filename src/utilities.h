#include <iostream>
#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <algorithm>
using namespace std;

std::vector<int> diff_union(std::vector<int> A, std::vector<int> B, std::vector<int> C);
std::vector<int> max_k(Eigen::VectorXd L, int k);
std::vector<int> min_k(Eigen::VectorXd L, int k);
Eigen::VectorXi find_ind(std::vector<int> L, Eigen::VectorXi& index, Eigen::VectorXi& gsize, int N, int p);
Eigen::MatrixXd X_seg(Eigen::MatrixXd& X, int n, Eigen::VectorXi& ind);
std::vector<int> vec_seg(std::vector<int> ind, std::vector<int> L);
std::vector<int> Ac(std::vector<int> A, int N);
std::vector<int> warm_start_gs(std::vector<int> V, Eigen::VectorXd inital, int k1, int k2);
  

