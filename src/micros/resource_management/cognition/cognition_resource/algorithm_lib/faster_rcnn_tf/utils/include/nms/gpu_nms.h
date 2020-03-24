#include <iostream>
#include <vector>
#include <algorithm>
#include <unsupported/Eigen/CXX11/Tensor>

typedef Eigen::Tensor<float, 2, Eigen::RowMajor> Tensor2f;
typedef Eigen::Tensor<int, 1, Eigen::RowMajor> Tensor1i;

void _nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id);

Tensor1i gpu_nms(Tensor2f dets, float thresh, int device_id = 0);
