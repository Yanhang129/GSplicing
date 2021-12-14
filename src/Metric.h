#ifndef SRC_METRICS_H
#define SRC_METRICS_H

#include "Data.h"
#include "Algorithm.h"
#include <vector>
#include <random>
#include <algorithm>
// [[Rcpp::plugins("cpp11")]]

class Metric {
public:
  bool is_cv;
  int K;
  int ic_type;
  Eigen::MatrixXd cv_initial_model_param;
  std::vector<Eigen::VectorXi> train_mask_list;
  std::vector<Eigen::VectorXi> test_mask_list;
  Metric() = default;
  
  Metric(int ic_type, bool is_cv, int K = 0) {
    this->is_cv = is_cv;
    this->ic_type = ic_type;
    this->K = K;
  };
  
  void set_cv_train_test_mask(int n) {
    Eigen::VectorXi index_list(n);
    std::vector<int> index_vec((unsigned int) n);
    for (int i = 0; i < n; i++) {
      index_vec[i] = i;
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(index_vec.begin(), index_vec.end(), g);
    
    for (int i = 0; i < n; i++) {
      index_list(i) = index_vec[i];
    }
    
    Eigen::VectorXd loss_list(this->K);
    std::vector<Eigen::VectorXi> group_list((unsigned int) this->K);
    int group_size = int(n / this->K);
    for (int k = 0; k < (this->K - 1); k++) {
      group_list[k] = index_list.segment(int(k * group_size), group_size);
    }
    group_list[this->K - 1] = index_list.segment(int((this->K - 1) * group_size),
                                                 n - int(int(this->K - 1) * group_size));
    
    // cv train-test partition:
    std::vector<Eigen::VectorXi> train_mask_list_tmp((unsigned int) this->K);
    std::vector<Eigen::VectorXi> test_mask_list_tmp((unsigned int) this->K);
    for (int k = 0; k < this->K; k++) {
      int train_x_size = n - group_list[k].size();
      // get train_mask
      Eigen::VectorXi train_mask(train_x_size);
      int i = 0;
      for (int j = 0; j < this->K; j++) {
        if (j != k) {
          for (int s = 0; s < group_list[j].size(); s++) {
            train_mask(i) = group_list[j](s);
            i++;
          }
        }
      }
      train_mask_list_tmp[k] = train_mask;
      test_mask_list_tmp[k] = group_list[k];
    }
    this->train_mask_list = train_mask_list_tmp;
    this->test_mask_list = test_mask_list_tmp;
  };
  
  virtual double test_loss(Algorithm *algorithm, Data &data) = 0;
  
  virtual double train_loss(Algorithm *algorithm, Data &data) = 0;
  
  virtual double ic(Algorithm *algorithm, Data &data) = 0;
};

class LmMetric : public Metric {
public:
  
  LmMetric(int ic_type, bool is_cv, int K = 0) : Metric(ic_type, is_cv, K) {};
  
  double train_loss(Algorithm *algorithm, Data &data) {
    return (data.y - data.x * algorithm->get_beta()).array().square().sum() / (data.get_n());
  }
  
  double test_loss(Algorithm *algorithm, Data &data) {
    if (!this->is_cv) {
      return (data.y - data.x * algorithm->get_beta()).array().square().sum() / (data.get_n());
    } else {
      int k;
      int p = data.get_p();
      algorithm->update_cv_label(2);
      double coef_init = 0.0;
      Eigen::VectorXd beta_init = Eigen::VectorXd::Zero(p);
      Eigen::VectorXd inital = algorithm->inital_screening(data.x, data.y, beta_init, coef_init, data.weight,
                                                           data.g_index, data.g_size, data.g_num);
      vector<int> A0 = max_k(inital, algorithm->get_model_size());
      vector<int> A1(A0);
      Eigen::VectorXd loss_list(this->K);
      
      for (k = 0; k < this->K; k++) {
        int test_size = this->test_mask_list[k].size();
        Eigen::MatrixXd test_x(test_size, p);
        Eigen::VectorXd test_y(test_size);
        Eigen::VectorXd test_weight(test_size);
        
        for (int i = 0; i < test_size; i++) {
          test_x.row(i) = data.x.row(this->test_mask_list[k](i));
          test_y(i) = data.y(this->test_mask_list[k](i));
          test_weight(i) = data.weight(this->test_mask_list[k](i));
        };
        algorithm->update_train_mask(this->train_mask_list[k]);
        algorithm->fit(A0);
        if (!algorithm->get_warm_start()) {
          A0 = A1;
        }
        loss_list(k) = (test_y - test_x * algorithm->get_beta()).array().square().sum() / double(2 * test_size);
      }
      return loss_list.mean();
    }
  };
  
  double ic(Algorithm *algorithm, Data &data) {
    if (this->is_cv) {
      return this->test_loss(algorithm, data);
    } else {
      if (ic_type == 1) {
        return double(data.get_n()) * log(this->train_loss(algorithm, data)) +
          2.0 * algorithm->get_group_df();
      } else if (ic_type == 2) {
        return double(data.get_n()) * log(this->train_loss(algorithm, data)) +
          log(double(data.get_n())) * algorithm->get_group_df();
      } else if (ic_type == 3) {
        return double(data.get_n()) * log(this->train_loss(algorithm, data)) +
          0.7*log(double(data.get_g_num())) * log(log(double(data.get_n()))) * algorithm->get_group_df();
      }  else return 0;
    }
  }
};

#endif //SRC_METRICS_H
