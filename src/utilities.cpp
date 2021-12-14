#include "utilities.h"
#include <algorithm>
#include <vector>
#include <iostream>
#include <Rcpp.h>
#include <RcppEigen.h>
using namespace std;

std::vector<int> diff_union(std::vector<int> A, std::vector<int> B, std::vector<int> C) 
{
  unsigned int k;
  for (unsigned int i=0;i<A.size();i++) {
    for (k=0;k<B.size();k++) {
      if (A[i] == B[k]) {
        A.erase(A.begin()+i);
        i--;
        break;
      }
    }
  }
  for (k=0;k<C.size();k++) {
    A.push_back(C[k]);
  }
  sort(A.begin(), A.end());
  return A;
}

std::vector<int> max_k(Eigen::VectorXd L, int k)
{
  std::vector<int> vec(k);
  for (int i=0;i<k;i++) 
  {
    L.maxCoeff(&vec[i]);
    L(vec[i])=-1;
  }
  return vec;
}

std::vector<int> min_k(Eigen::VectorXd L, int k)
{
  std::vector<int> vec(k);
  for (int i=0;i<k;i++) 
  {
    L.minCoeff(&vec[i]);
    L(vec[i])=1e100;
  }
  return vec;
}

std::vector<int> Ac(std::vector<int> A, int N)
{
  int A_size = A.size();
  int temp = 0;
  int j = 0;
  if (A_size != 0) {
    bool label;
    std::vector<int> vec;
    for (int i=0;i<N;i++) {
      label = FALSE;
      for (;j<A_size;j++) {
        if (i == A[j]) {
          label = TRUE;
          temp++;
          break;
        }
      }
      j = temp;
      if (label == TRUE) {
        continue;
      }
      else {
        vec.push_back(i);
      }
    }
    return vec;
  }
  
  else {
    std::vector<int> vec(N);
    for (int i=0;i<N;i++) {
      vec[i] = i;
    }
    return vec;
  }
}

Eigen::VectorXi find_ind(std::vector<int> L, Eigen::VectorXi& index, Eigen::VectorXi& gsize, int p, int N)
{
  unsigned int J = N;
  if (L.size() == J) {
    return Eigen::VectorXi::LinSpaced(p, 0, p-1);
  }
  else 
  {
    int mark = 0;
    Eigen::VectorXi ind = Eigen::VectorXi::Zero(p);
    for (unsigned int i=0;i<L.size();i++) {
      ind.segment(mark, gsize(L[i])) = Eigen::VectorXi::LinSpaced(gsize(L[i]), index(L[i]), index(L[i])+gsize(L[i])-1);
      mark = mark + gsize(L[i]);
    }
    return ind.head(mark);
  }
}

Eigen::MatrixXd X_seg(Eigen::MatrixXd& X, int n, Eigen::VectorXi& ind) 
{
  Eigen::MatrixXd X_new(n, ind.size());
  for (int k=0;k<ind.size();k++) {
    X_new.col(k) = X.col(ind[k]);
  }
  return X_new;
}

std::vector<int> vec_seg(std::vector<int> L, std::vector<int> ind) {
  std::vector<int> vec(ind.size());
  for (unsigned int i=0;i<ind.size();i++) {
    vec[i] = L[ind[i]];
  }
  return vec;
}

std::vector<int> warm_start_gs(std::vector<int> V, Eigen::VectorXd inital, int k1, int k2)
{
  if (k1 > k2)
  {
    std::vector<int> temp = max_k(inital, k1-k2);
    V.insert(V.end(), temp.begin(), temp.end());
    sort(V.begin(), V.end());
    return V;
  } 
  else if (k1 == k2) 
  {
    return V ;  
  } else {
    V = min_k(inital, k1);
    sort(V.begin(), V.end());
    return V;
  }
}
