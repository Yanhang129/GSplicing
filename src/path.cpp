#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Rcpp;
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppEigen)]]
#include <iostream>
#include "Data.h"
#include "Algorithm.h"
#include "Metric.h"
#include "utilities.h"

using namespace Eigen;
using namespace std;

List sequential_path(Data &data, Algorithm *algorithm, Metric *metric, Eigen::VectorXi sequence, double coef_init, double tau)
{
  int p = data.p;
  int N = data.g_num;
  int sequence_size = sequence.size();
  List A_out(sequence_size);
  Eigen::VectorXd iter_sequence = Eigen::VectorXd::Zero(sequence_size);
  Eigen::VectorXd ic_sequence = Eigen::VectorXd::Zero(sequence_size);
  Eigen::VectorXd loss_sequence = Eigen::VectorXd::Zero(sequence_size);
  Eigen::MatrixXd beta_matrix= Eigen::MatrixXd::Zero(p, sequence_size);
  Eigen::VectorXd coef0_sequence = Eigen::VectorXd::Zero(sequence_size);
  Eigen::VectorXd beta_init = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd inital = algorithm->inital_screening(data.x, data.y, beta_init, coef_init, data.weight,
                                                       data.g_index, data.g_size, data.g_num);
  vector<int> A_inital;
  algorithm->update_tau(tau);
  int i;
  for (i = 0; i < sequence_size; i++)
  {
    if (algorithm->warm_start && i>0)
    {
      int diff = sequence(i) - sequence(i-1);
      Eigen::VectorXd beta_temp = beta_matrix.col(i-1);
      Eigen::VectorXd temp = algorithm->inital_screening(data.x, data.y, beta_temp, coef0_sequence(i-1), data.weight, data.g_index, data.g_size, data.g_num);
      vector<int> A_new = max_k(temp, diff);
      A_inital.insert(A_inital.end(), A_new.begin(), A_new.end());
      sort(A_inital.begin(), A_inital.end());
    } else {
      A_inital = max_k(inital, sequence(i));
    }
    algorithm->update_cv_label(1);
    algorithm->update_model_size(sequence(i));
    algorithm->fit(A_inital);
    A_out[i] = algorithm->get_A_out();
    loss_sequence(i) = metric->train_loss(algorithm, data);
    if (i>0 && abs(loss_sequence(i-1)-loss_sequence(i))/loss_sequence(i-1)>0.95)
    {
      loss_sequence(i) = 0.0;
      break;
    }
    beta_matrix.col(i) = algorithm->get_beta();
    coef0_sequence(i) = algorithm->get_coef();
    iter_sequence(i) = algorithm->get_l();
    ic_sequence(i) = metric->ic(algorithm, data);
  }
  
  
  for (int i=0;i<N;i++) {
    beta_matrix.block(data.g_index(i), 0, data.g_size(i), sequence_size) = data.mat[i] * beta_matrix.block(data.g_index(i), 0, data.g_size(i), sequence_size);
  }
  coef0_sequence = data.y_mean*Eigen::VectorXd::Ones(sequence_size) - beta_matrix.transpose()*data.x_mean;
  
  int min_ic = 0;
  (ic_sequence.head(i)).minCoeff(&min_ic);
  
  return List::create(Named("beta") = beta_matrix,
                      Named("intercept") = coef0_sequence,
                      Named("A_out") = A_out,
                      Named("best_group") = A_out[min_ic],
                                                 Named("best_model_size") = min_ic+1,
                                                 Named("loss") = loss_sequence,
                                                 Named("ic") = ic_sequence);
}


List gs_path(Data &data, Algorithm *algorithm, Metric *metric, int s_min, int s_max, double coef_init, double tau)
{
  int p = data.get_p();
  int N = data.get_g_num();
  int T1, T2, TL = s_min, TR = s_max;
  double devL = 0.0, dev_warm = 0.0, ic_warm = 0.0, devR, coefL, coefR, coef_warm, icL, icR;
  std::vector<int> AL;
  std::vector<int> AR;
  std::vector<int> A_temp;
  std::vector<int> A_temp2;
  std::vector<int> A_warm;
  Eigen::VectorXd temp = Eigen::VectorXd::Zero(N);
  Eigen::VectorXd beta_warm = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd betaL = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd betaR = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd beta_init = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd inital = algorithm->inital_screening(data.x, data.y, beta_init, coef_init, data.weight,
                                                       data.g_index, data.g_size, data.g_num);
  algorithm->update_tau(tau);
  algorithm->update_cv_label(1);
  T1 = round(0.618*TL+0.382*TR);
  T2 = round(0.382*TL+0.618*TR);
  AL = max_k(inital, T1);
  sort(AL.begin(), AL.end());
  AR = max_k(inital, T2);
  sort(AR.begin(), AR.end());
  algorithm->update_model_size(T1);
  algorithm->fit(AL);
  coefL = algorithm->get_coef();
  betaL = algorithm->get_beta();
  devL = metric->train_loss(algorithm, data);
  icL = metric->ic(algorithm, data);
  
  algorithm->update_model_size(T2);
  algorithm->fit(AR);
  coefR = algorithm->get_coef();
  betaR = algorithm->get_beta();
  devR = metric->train_loss(algorithm, data);
  icR = metric->ic(algorithm, data);
  for (int i=0;i<20;i++) {
    if (icL > icR && devR > 1e-3) {
      TL = T1;
      T1 = T2;
      T2 = round(0.382*TL+0.618*TR);
      A_warm = AL;
      coef_warm = coefL;
      beta_warm = betaL;
      dev_warm = devL;
      ic_warm = icL;
      if (TR-TL <= 4)
      {
        break;
      }
      A_temp2 = AR;
      temp = algorithm->inital_screening(data.x, data.y, betaL, coefL, data.weight, data.g_index, data.g_size, data.g_num);
      A_temp = max_k(temp, T2-TL);
      AR = AL;
      AR.insert(AR.end(), A_temp.begin(), A_temp.end());
      sort(AR.begin(), AR.end());
      icL = icR;
      betaL = betaR;
      coefL = coefR;
      devL = devR;
      AL = A_temp2;
      algorithm->update_model_size(T2);
      algorithm->fit(AR);
      coefR = algorithm->get_coef();
      betaR = algorithm->get_beta();
      devR = metric->train_loss(algorithm, data);
      icR = metric->ic(algorithm, data);
    } else {
      TR = T2;
      T2 = T1;
      T1 = round(0.618*TL+0.382*TR);
      if (TR-TL <= 4)
      {
        break;
      }
      AR = AL;
      temp = algorithm->inital_screening(data.x, data.y, betaL, coefL, data.weight, data.g_index, data.g_size, data.g_num);
      AL = min_k(temp, T1);
      sort(AL.begin(), AL.end());
      icR = icL;
      betaR = betaL;
      coefR = coefL;
      devR = devL;
      algorithm->update_model_size(T1);
      algorithm->fit(AL);
      coefL = algorithm->get_coef();
      betaL = algorithm->get_beta();
      devL = metric->train_loss(algorithm, data);
      icL = metric->ic(algorithm, data);
    }
  }
  
  List A_out(TR-TL+1);
  Eigen::VectorXd ic_sequence = Eigen::VectorXd::Zero(TR-TL+1);
  Eigen::VectorXd loss_sequence = Eigen::VectorXd::Zero(TR-TL+1);
  Eigen::MatrixXd beta_matrix= Eigen::MatrixXd::Zero(p, TR-TL+1);
  Eigen::VectorXd coef0_sequence = Eigen::VectorXd::Zero(TR-TL+1);
  if (TL == 1) {
    AL = max_k(inital, 1);
    algorithm->update_model_size(1);
    algorithm->fit(AL);
    A_out[0] = algorithm->get_A_out();
    loss_sequence(0) = metric->train_loss(algorithm, data);
    ic_sequence(0) = metric->ic(algorithm, data);
    betaL = algorithm->get_beta();
    coefL = algorithm->get_coef();
    beta_matrix.col(0) = betaL;
    coef0_sequence(0) = coefL;
    temp = algorithm->inital_screening(data.x, data.y, betaL, coefL, data.weight, data.g_index, data.g_size, data.g_num);
    for (int i=2;i<=(TR-TL+1);i++)
    {
      if (algorithm->warm_start)
      {
        AR = AL;
        A_temp = max_k(temp, i-AL.size());
        AR.insert(AR.end(), A_temp.begin(), A_temp.end());
        sort(AR.begin(), AR.end());
      } else {
        AR = max_k(inital, i);
      }
      algorithm->update_model_size(i);
      algorithm->fit(AR);
      A_out[i-1] = algorithm->get_A_out();
      loss_sequence(i-1) = metric->train_loss(algorithm, data);
      ic_sequence(i-1) = metric->ic(algorithm, data);
      beta_matrix.col(i-1) = algorithm->get_beta();
      coef0_sequence(i-1) = algorithm->get_coef();
    }
  }
  else {
    A_temp= A_warm;
    for (unsigned int i=0;i<A_temp.size();i++)
    {
      A_temp[i] = A_warm[i] + 1;
    }
    A_out(0) = A_temp;
    ic_sequence(0) = ic_warm;
    loss_sequence(0) = dev_warm;
    beta_matrix.col(0) = beta_warm;
    coef0_sequence(0) = coef_warm;
    temp = algorithm->inital_screening(data.x, data.y, beta_warm, coef_warm, data.weight, data.g_index, data.g_size, data.g_num);
    for (int i=TL+1;i<=TR;i++)
    {
      if (algorithm->warm_start)
      {
        AR = A_warm;
        A_temp = max_k(temp, i-TL);
        AR.insert(AR.end(), A_temp.begin(), A_temp.end());
        sort(AR.begin(), AR.end());
      } else {
        AR = max_k(inital, i);
      }
      algorithm->update_model_size(i);
      algorithm->fit(AR);
      A_out[i-TL] = algorithm->get_A_out();
      loss_sequence(i-TL) = metric->train_loss(algorithm, data);
      ic_sequence(i-TL) = metric->ic(algorithm, data);
      beta_matrix.col(i-TL) = algorithm->get_beta();
      coef0_sequence(i-TL) = algorithm->get_coef();
    }
  }
  int min_ic = 0;
  ic_sequence.minCoeff(&min_ic);
  Eigen::VectorXd beta_out = Eigen::VectorXd::Zero(p);
  double intercept = 0.0;
  for (int i=0;i<N;i++) {
    beta_out.segment(data.g_index(i), data.g_size(i)) = data.mat[i] * beta_matrix.block(data.g_index(i), min_ic, data.g_size(i), 1);
  }
  intercept = data.y_mean - beta_out.dot(data.x_mean);
  return List::create(Named("beta")=beta_out, Named("intercept")=intercept, Named("best_model_size")=TL+min_ic, Named("best_group")=A_out[min_ic], Named("ic")=ic_sequence[min_ic], Named("loss")=loss_sequence[min_ic]);
}
