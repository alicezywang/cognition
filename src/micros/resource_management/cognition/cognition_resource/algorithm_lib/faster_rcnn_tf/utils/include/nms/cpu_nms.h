#include <iostream>
#include <vector>
#include <algorithm>
#include <unsupported/Eigen/CXX11/Tensor>

typedef Eigen::Tensor<float, 2, Eigen::RowMajor> Tensor2f;
typedef Eigen::Tensor<int, 1, Eigen::RowMajor> Tensor1i;

Tensor1i cpu_nms(Tensor2f dets, float thresh);

Tensor1i cpu_nms_old(Tensor2f dets, float thresh);
