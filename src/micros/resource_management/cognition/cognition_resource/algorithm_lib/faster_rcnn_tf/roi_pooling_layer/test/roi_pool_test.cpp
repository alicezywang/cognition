#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/framework/gradients.h>
#include <tensorflow/core/framework/tensor.h>
#include "roi_pool.h"

using namespace tensorflow;
using namespace tensorflow::ops;

int number_stored = 1;

int get_number()
{
    number_stored = (number_stored * 101 + 59) % 32768;
    return number_stored;
}

int main()
{
    int i, j, k;
    std::vector<Tensor> outputs;
    std::vector<Output> diff;
    Scope root = Scope::NewRootScope();
    ClientSession session(root);

    Tensor data(DataType::DT_FLOAT, {32,100,100,3});
    auto data_pointer = data.tensor<float, 4>();
    for(i=0; i<32; i++)
    {
        for(j=0; j<100; j++)
        {
            for(k=0; k<100; k++)
            {
                data_pointer(i, j, k, 0) = (float)get_number() / 32768.0f;
                data_pointer(i, j, k, 1) = (float)get_number() / 32768.0f;
                data_pointer(i, j, k, 2) = (float)get_number() / 32768.0f;
            }
        }
    }

    Input rois = Const(root, {{0.0f,10.0f,10.0f,20.0f,20.0f},{31.0f,30.0f,30.0f,40.0f,40.0f}});
    Variable w(root, {3,3,3,1}, DataType::DT_FLOAT);
    TF_CHECK_OK(session.Run({Assign(root, w, Tensor(DataType::DT_FLOAT, {3,3,3,1}))}, NULL));

    Conv2D h(root, data, w, {1,1,1,1}, "SAME");
    Tensor y_data(DataType::DT_FLOAT, {2,6,6,1});
    for(i=0; i<6; i++)
    {
        for(j=0; j<6; j++)
        {
            y_data.tensor<float, 4>()(0, i, j, 0) = 1.0f;
            y_data.tensor<float, 4>()(1, i, j, 0) = 1.0f;
        }
    }

    RoiPool y(root, h, rois, 6, 6, 1.0f/3.0f);
    Mean cost(root, Square(root, Sub(root, y.top_data, y_data)), {0,1,2,3});
    TF_CHECK_OK(session.Run({cost}, &outputs));

    TF_CHECK_OK(AddSymbolicGradients(root, {cost}, {w}, &diff));
    ApplyGradientDescent gd(root, w, Const(root, 0.5f), diff[0]);

    for(i=0; i<10; i++)
    {
        std::cout << (i+1) << std::endl << std::endl;
        TF_CHECK_OK(session.Run({gd}, NULL));
        TF_CHECK_OK(session.Run({w}, &outputs));
        int n = outputs[0].shape().num_elements();
        auto output1 = outputs[0].flat<float>().data();
        for(j=0; j<n; j++)
        {
            std::cout << std::fixed << output1[j] << std::endl;
        }
        std::cout << std::endl;
        TF_CHECK_OK(session.Run({y.top_data}, &outputs));
        n = outputs[0].shape().num_elements();
        auto output2 = outputs[0].flat<float>().data();
        for(j=0; j<n; j++)
        {
            std::cout << std::fixed << output2[j] << std::endl;
        }
        std::cout << std::endl;
    }
    return 0;
}
