# tensor train regression

Code for the paper:
Yipeng Liu, Jiani Liu, Ce Zhu, "Low Rank Tensor Train Coe cient Array Estimation for Tensor-on-Tensor Regression," IEEE Transactions on Neural Networks and Learning Systems, 2020.  DoI: doi.org/10.1109/TNNLS.2020.2967022 

abstract: The tensor-on-tensor regression can predict a tensor from a tensor, which generalizes most previous multi-linear regression approaches, including methods to predict a scalar from a tensor, and a tensor from a scalar. However, the coefficient array could be much higher dimensional due to both high order predictors and responses in this generalized way. Compared with the current low CP rank approximation based method, the low tensor train approximation can further improve the stability and efficiency of the high or even ultra-high dimensional coefficient array estimation. In the proposed low tensor train rank coefficient array estimation for tensor-on-tensor regression, we adopt a tensor train rounding procedure to obtain adaptive ranks, instead of selecting ranks by experience. Besides, an $ \ell _2 $ constraint is imposed to avoid overfitting. The hierarchical alternating least squares is used to solve the optimization problem. Numerical experiments on a synthetic dataset and two real-life datasets demonstrate that the proposed method outperforms the state-of-the-art methods in terms of prediction accuracy with comparable computational complexity, and the proposed method is more computationally efficient when the data is high dimensional with small size in each mode.


## Requirements

MATLAB Tensor Toolbox 2.6 (http://www.sandia.gov/~tgkolda/TensorToolbox/)

TT-Toolbox (TT=Tensor Train) Version 2.2 (http://spring.inm.ras.ru/osel)


## Contents

The main algorithms are in /tools. 

The demo folder contains files for reproducing the synthetic experiments from the paper.

