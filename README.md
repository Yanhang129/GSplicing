# GSplicing: Best Subset of Groups Selection in Linear Model


## Quick start

Install the stable version of R-package from Github with:

```r
devtools::install_github("Weiniily/GSplicing")
```

Best subset of groups selection for group linear regression on a simulated dataset in R:

```r
library(GSplicing)
n <- 200
J <- 100
k <- 5
model.size <- 5
data <- generate.group(n, J, k, model.size)
fit <- gsplicing(data$x, data$y, data$group)
all(fit$best_group == data$true.group)
```

## References

- Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, and Xueqin Wang (2020). A polynomial algorithm for best-subset selection problem. Proceedings of the National Academy of Sciences, 117(52):33117-33123.

- Yanhang Zhang, Junxian Zhu, Jin Zhu, and Xueqin Wang (2021). Certifiably Polynomial Algorithm for Best Group Subset Selection. arXiv preprint arXiv:2104.12576.
