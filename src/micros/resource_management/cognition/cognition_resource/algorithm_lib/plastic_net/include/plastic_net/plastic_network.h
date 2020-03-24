#ifndef NETWORK_OMNIGLOT_H_
#define NETWORK_OMNIGLOT_H_

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>

using tensorflow::Input;
using tensorflow::Output;
using tensorflow::OutputHash;
using tensorflow::Tensor;
using tensorflow::DT_FLOAT;
using namespace tensorflow::ops;

/**
 * @brief The Convolution class
 */
class Convolution
{
private:
    int c_in, c_out, k_size, stride;
    bool initialized;
public:
    Output weight, bias;
    Convolution(int c_in = 1, int c_out = 1, int k_size = 3, int stride = 2);
    /**
     * @brief operator ()
     * @param scope
     * @param session
     * @param inputx
     * @param feed_map
     * @return
     */
    Output operator()(const tensorflow::Scope &scope, tensorflow::ClientSession &session, Input inputx, const std::unordered_map<Output, Input::Initializer, OutputHash> &feed_map);
};

/**
 * @brief The Network class
 */
class Network
{
private:
    bool is_oja, is_free, is_tanh, is_relu;
    tensorflow::Scope scope;
    tensorflow::ClientSession &session;
    void initParams();
public:
    Convolution cv1, cv2, cv3, cv4;
    Output w, alpha, eta, out, inputs, labels;
    Network(const tensorflow::Scope &scope, tensorflow::ClientSession &session);
    ~Network();
    /**
     * @brief forward
     * @param inputx
     * @param input_label
     * @param nbf
     * @param nbclasses
     * @param hebb
     * @param feed_map
     * @return
     */
    Output forward(Input inputx, Input input_label, int nbf, int nbclasses, Output &hebb, const std::unordered_map<Output, Input::Initializer, OutputHash> &feed_map);
};

#endif
