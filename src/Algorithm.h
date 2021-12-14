#ifndef SRC_ALGORITHM_H
#define SRC_ALGORITHM_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <algorithm>
#include <vector>
#include "Data.h"
#include "utilities.h"
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppEigen)]]
using namespace std;

class Algorithm {
public:
  Data data;
  int group_df;
  int model_size;
  Eigen::VectorXi train_mask;
  int max_iter;
  int exchange_num;
  bool warm_start;
  Eigen::VectorXd beta;
  double coef;
  double loss;
  Eigen::VectorXi A_out;
  int l;
  int model_fit_max;
  int cv_label;
  double tau;
  
  Algorithm() = default;
  
  Algorithm(Data &data, int max_iter, int exchange_num)
  {
    this->data = data;
    this->max_iter = max_iter;
    this->A_out = Eigen::VectorXi::Zero(data.get_p());
    this->coef = 0.0;
    this->beta = Eigen::VectorXd::Zero(data.get_p());
    this->warm_start = true;
    this->exchange_num = exchange_num;
  };
  
  void set_warm_start(bool warm_start) {
    this->warm_start = warm_start;
  };
  
  void update_cv_label(int cv_label) {
    this->cv_label = cv_label;
  };
  
  void update_tau(double tau) {
    this->tau = tau;
  }
  
  void update_group_df(int group_df) {
    this->group_df = group_df;
  };
  
  void update_model_size(int model_size) {
    this->model_size = model_size;
  }
  
  
  void update_train_mask(Eigen::VectorXi train_mask) {
    this->train_mask = train_mask;
  }
  
  void update_exchange_num(int exchange_num) {
    this->exchange_num = exchange_num;
  };
  
  bool get_warm_start() {
    return this->warm_start;
  }
  
  double get_loss() {
    return this->loss;
  }
  
  int get_group_df() {
    return this->group_df;
  };
  
  int get_model_size() {
    return this->model_size;
  }
  
  Eigen::VectorXd get_beta() {
    return this->beta;
  }
  
  double get_coef() {
    return this->coef;
  }
  
  Eigen::VectorXi get_A_out() {
    return this->A_out;
  };
  
  int get_l() {
    return this->l;
  }
  
  void fit(vector<int>& A0) {
    Eigen::MatrixXd x;
    Eigen::VectorXd y;
    Eigen::VectorXd weight;
    int n;
    int p = data.get_p();
    if (this->cv_label != 1)
    {
      n = this->train_mask.size();
      x = Eigen::MatrixXd::Zero(n, p);
      y = Eigen::VectorXd::Zero(n);
      weight = Eigen::VectorXd::Zero(n);
      for (int i = 0; i < n; i++) {
        x.row(i) = data.x.row(this->train_mask(i));
        y(i) = data.y(this->train_mask(i));
        weight(i) = data.weight(this->train_mask(i));
      }
    } else {
      x = data.x;
      y = data.y;
      weight = data.weight;
      n = data.n;
    }
    
    int T0 = this->model_size;
    int N = data.get_g_num();
    Eigen::VectorXi index = data.get_g_index();
    Eigen::VectorXi gsize = data.get_g_size();
    Eigen::VectorXd d = Eigen::VectorXd::Zero(p);
    vector<int> A1(A0);
    Eigen::VectorXi A_ind = find_ind(A0, index, gsize, p, N);
    Eigen::MatrixXd X_A = X_seg(x, n, A_ind);
    Eigen::VectorXd beta_A = Eigen::VectorXd::Zero(A_ind.size());
    this->primary_model_fit(X_A, y, weight, beta_A, this->coef);
    this->beta = Eigen::VectorXd::Zero(p);
    for (int k=0;k<A_ind.size();k++) {
      this->beta(A_ind(k)) = beta_A(k);
    }
    vector<int> I0(Ac(A0, N));
    vector<int> I1(I0);
    Eigen::VectorXi I_ind = find_ind(I0, index, gsize, p, N);
    Eigen::MatrixXd X_I = X_seg(x, n, I_ind);
    Eigen::VectorXd d_I = this->dual(X_I, X_A, y, beta_A, this->coef, weight, n);
    for (int k=0;k<I_ind.size();k++) {
      d(I_ind(k)) = d_I(k);
    }
    
    for(this->l=0;this->l<this->max_iter;l++) {
      if (min(T0, N-T0) <= this->exchange_num) {
        this->get_A(x, y, A0, I0, min(T0, N-T0), this->beta, this->coef, d, weight, index, gsize, n, p, N, T0*this->tau);
      }
      else {
        this->get_A(x, y, A0, I0, this->exchange_num, this->beta, this->coef, d, weight, index, gsize, n, p, N, T0*this->tau);
      }
      
      if (A0 == A1) {
        l++;
        break;
      } else {
        A1 = A0;
        I1 = I0;
      }
    }
    this->group_df = 0;
    this->A_out = Eigen::VectorXi::Zero(this->model_size);
    for (unsigned int i=0;i<A0.size();i++)
    {
      this->A_out[i] = A0[i] + 1;
      this->group_df = this->group_df+data.g_size(A0[i]);
    }
  }
  
  virtual Eigen::VectorXd inital_screening(Eigen::MatrixXd& X, Eigen::VectorXd& y, Eigen::VectorXd& beta, double& coef0, Eigen::VectorXd& weights,
                                           Eigen::VectorXi& index, Eigen::VectorXi& gsize, int& N)=0;
  
  virtual void primary_model_fit(Eigen::MatrixXd& X, Eigen::VectorXd& y, Eigen::VectorXd& weights, Eigen::VectorXd &beta, double &coef0)=0;
  
  virtual Eigen::VectorXd dual(Eigen::MatrixXd& XI, Eigen::MatrixXd& XA, Eigen::VectorXd& y, Eigen::VectorXd& beta, double& coef0, Eigen::VectorXd& weight, int n)=0;
  
  virtual void get_A(Eigen::MatrixXd& X, Eigen::VectorXd& y, vector<int>& A, vector<int>& I, int C_max, Eigen::VectorXd& beta, double& coef0, Eigen::VectorXd& d, Eigen::VectorXd& weights,
                     Eigen::VectorXi& index, Eigen::VectorXi& gsize, int n, int p, int N, double tau)=0;
};

class GSplicingLm : public Algorithm {
public:
  GSplicingLm(Data &data, int max_iter, int exchange_num) : Algorithm(data, max_iter, exchange_num){};
  
  Eigen::VectorXd inital_screening(Eigen::MatrixXd& X, Eigen::VectorXd& y, Eigen::VectorXd& beta, double& coef0, Eigen::VectorXd& weights,
                                   Eigen::VectorXi& index, Eigen::VectorXi& gsize, int& N)
  {
    Eigen::VectorXd res = y-X*beta;
    Eigen::VectorXd Xy = X.transpose()*res;
    Eigen::VectorXd inital = Eigen::VectorXd::Zero(N);
    for(int i=0;i<N;i++){
      inital(i) = (Xy.segment(index(i), gsize(i)).squaredNorm());
    }
    return inital;
  }
  
  Eigen::VectorXd dual(Eigen::MatrixXd& XI, Eigen::MatrixXd& XA, Eigen::VectorXd& y, Eigen::VectorXd& beta, double& coef0, Eigen::VectorXd& weight, int n)
  {
    return XI.transpose()*(y-XA*beta)/n;
  }
  
  void get_A(Eigen::MatrixXd& X, Eigen::VectorXd& y, vector<int>& A, vector<int>& I, int C_max, Eigen::VectorXd& beta, double& coef0, Eigen::VectorXd& d, Eigen::VectorXd& weights,
             Eigen::VectorXi& index, Eigen::VectorXi& gsize, int n, int p, int N, double tau)
  {
    double L0 = (y-X*beta).squaredNorm();
    double L1;
    int A_size = A.size();
    int I_size = I.size();
    Eigen::VectorXd beta_A = Eigen::VectorXd::Zero(A_size);
    Eigen::VectorXd d_I = Eigen::VectorXd::Zero(I_size);
    for(int i=0;i<A_size;i++) {
      beta_A(i) = beta.segment(index(A[i]), gsize(A[i])).squaredNorm();
    }
    for(int i=0;i<I_size;i++) {
      d_I(i) = d.segment(index(I[i]), gsize(I[i])).squaredNorm();
    }
    vector<int>s1(vec_seg(A, min_k(beta_A, C_max)));
    vector<int>s2(vec_seg(I, max_k(d_I, C_max)));
    vector<int>A_c(A_size);
    for (int k=C_max;k>=1;k--) {
      A_c = diff_union(A, s1, s2);
      Eigen::VectorXi A_ind = find_ind(A_c, index, gsize, p, N);
      Eigen::MatrixXd X_A = X_seg(X, n, A_ind);
      Eigen::VectorXd beta_Ac = Eigen::VectorXd::Zero(A_ind.size());
      primary_model_fit(X_A, y, weights, beta_Ac, coef0);
      L1 = (y-X_A*beta_Ac).squaredNorm();
      if (L0 - L1 > tau) {
        A = A_c;
        I = Ac(A, N);
        beta = Eigen::VectorXd::Zero(p);
        d = X.transpose()*(y-X_A*beta_Ac);
        for (int i=0;i<A_ind.size();i++) {
          d(A_ind[i]) = 0;
          beta(A_ind[i]) = beta_Ac(i);
        }
        break;
      }
      else {
        s1.pop_back();
        s2.pop_back();
      }
    }
  };
  
  void primary_model_fit(Eigen::MatrixXd& X, Eigen::VectorXd& y, Eigen::VectorXd& weights, Eigen::VectorXd &beta, double &coef0)
  {
    beta = X.colPivHouseholderQr().solve(y);
  };
};

#endif //SRC_ALGORITHM_H

