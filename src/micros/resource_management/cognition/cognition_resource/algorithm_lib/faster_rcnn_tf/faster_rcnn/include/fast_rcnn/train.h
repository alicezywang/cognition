#ifndef TRAIN_H
#define TRAIN_H
#include <iostream>
#include <fstream>
#include <string>

#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/gradients.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/kernels/training_ops.h>
#include <tensorflow/core/kernels/training_op_helpers.h>

#include <vggnet_train.h>
#include <fast_rcnn/config.h>
#include <roi_data_layer/roi_data_layer.h>

namespace fast_rcnn {

using namespace tensorflow;

class SolverWrapper {

public:
    SolverWrapper(const Scope &scope, std::string output_dir);
    ~SolverWrapper();
    void snapshot(int iter);
    void train_model(int max_iters, int iter);

private:
    Scope scope;
    ClientSession session;
    vggnet_train net;
    std::vector<float> bbox_stds;
    std::vector<float> bbox_means;
    std::string output_dir;
    rdl::RoIDataLayer data_layer;
    Output _modified_smooth_l1(float sigma, Input bbox_pred, Input bbox_targets, Input bbox_inside_weights, Input bbox_outside_weights);
};

void train_net(const Scope &scope, const std::string &output_dir, const int iter, const int max_iters);

}

#endif
