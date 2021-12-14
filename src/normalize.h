#include <RcppEigen.h>

std::vector<Eigen::MatrixXd> Orthonormalize(Eigen::MatrixXd& X, Eigen::VectorXd& y, Eigen::VectorXi& index, Eigen::VectorXi& gsize, int n, int p, int N, Eigen::VectorXd& weights, Eigen::VectorXd& meanx, double& meany);
