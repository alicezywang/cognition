#include "plastic_net/plastic_net_model.h"
#include <ros/package.h>

using namespace machine_learning;

PlasticNetModel::PlasticNetModel(const tensorflow::Scope &scope): scope(scope), session(scope), net(this->scope, session)
{
    ml_datasets = ros::package::getPath("ml_datasets");
    pretrained_model = ros::package::getPath("pretrained_model");
}

PlasticNetModel::~PlasticNetModel()
{

}

void PlasticNetModel::restore(std::string filename)
{
    std::vector<std::string> temp_arr_slice(11);
    std::vector<Output> tensor_list;
    tensor_list.push_back(net.cv1.weight);
    tensor_list.push_back(net.cv2.weight);
    tensor_list.push_back(net.cv3.weight);
    tensor_list.push_back(net.cv4.weight);
    tensor_list.push_back(net.cv1.bias);
    tensor_list.push_back(net.cv2.bias);
    tensor_list.push_back(net.cv3.bias);
    tensor_list.push_back(net.cv4.bias);
    tensor_list.push_back(net.w);
    tensor_list.push_back(net.alpha);
    tensor_list.push_back(net.eta);
    temp_arr_slice[0] = "conv1/weight";
    temp_arr_slice[1] = "conv2/weight";
    temp_arr_slice[2] = "conv3/weight";
    temp_arr_slice[3] = "conv4/weight";
    temp_arr_slice[4] = "conv1/bias";
    temp_arr_slice[5] = "conv2/bias";
    temp_arr_slice[6] = "conv3/bias";
    temp_arr_slice[7] = "conv4/bias";
    temp_arr_slice[8] = "hyper/w";
    temp_arr_slice[9] = "hyper/alpha";
    temp_arr_slice[10] = "hyper/eta";
    for(int i=0; i<11; i++)
    {
        TF_CHECK_OK(session.Run({Assign(scope, tensor_list[i], tensorflow::ops::Restore(scope, filename, temp_arr_slice[i], DT_FLOAT))}, NULL));
    }
}

void PlasticNetModel::snapshot(int iter)
{
    const std::string filename = pretrained_model + "/TensorFlow/pnn/plastic_net_" + std::to_string(iter + 1) + std::string(".ckpt");
    Tensor arr_slice_tensor(tensorflow::DT_STRING, {11});
    auto temp_arr_slice = arr_slice_tensor.vec<std::string>();
    std::vector<Output> tensor_list;
    tensor_list.push_back(net.cv1.weight);
    tensor_list.push_back(net.cv2.weight);
    tensor_list.push_back(net.cv3.weight);
    tensor_list.push_back(net.cv4.weight);
    tensor_list.push_back(net.cv1.bias);
    tensor_list.push_back(net.cv2.bias);
    tensor_list.push_back(net.cv3.bias);
    tensor_list.push_back(net.cv4.bias);
    tensor_list.push_back(net.w);
    tensor_list.push_back(net.alpha);
    tensor_list.push_back(net.eta);
    temp_arr_slice(0) = "conv1/weight";
    temp_arr_slice(1) = "conv2/weight";
    temp_arr_slice(2) = "conv3/weight";
    temp_arr_slice(3) = "conv4/weight";
    temp_arr_slice(4) = "conv1/bias";
    temp_arr_slice(5) = "conv2/bias";
    temp_arr_slice(6) = "conv3/bias";
    temp_arr_slice(7) = "conv4/bias";
    temp_arr_slice(8) = "hyper/w";
    temp_arr_slice(9) = "hyper/alpha";
    temp_arr_slice(10) = "hyper/eta";
    TF_CHECK_OK(session.Run(std::unordered_map<Output, Input::Initializer, OutputHash>(), std::vector<Output>(), {Save(scope, filename, arr_slice_tensor, tensor_list)}, NULL));
    std::cout << "Wrote snapshot to: " << filename << std::endl;
}

void PlasticNetModel::train(int start, int end)
{
    srand(params.rngseed);
    ImgData img_data(ml_datasets + "/train_datasets/miniimagenet/train/");
    auto target = Placeholder(scope, DT_FLOAT, Placeholder::Shape({params.nbclesses}));
    auto ce1 = Negate(scope, Multiply(scope, target, Log(scope, net.out)));
    auto ce2 = Negate(scope, Multiply(scope, Sub(scope, 1.0f, target), Log(scope, Sub(scope, 1.0f, net.out))));
    auto loss = Mean(scope, Add(scope, ce1, ce2), {0});

    this->restore(pretrained_model + "/TensorFlow/pnn/plastic_net_" + std::to_string(start) + std::string(".ckpt"));

    std::vector<Output> variables = {net.cv1.weight, net.cv2.weight, net.cv3.weight, net.cv4.weight, net.cv1.bias, net.cv2.bias, net.cv3.bias, net.cv4.bias, net.w, net.alpha, net.eta};
    std::vector<Output> grads;
    std::vector<Output> train_ops;
    std::vector<Tensor> outputs;
    Output learning_rate = Placeholder(scope, DT_FLOAT, Placeholder::Shape({}));
    TF_CHECK_OK(AddSymbolicGradients(scope, {loss}, variables, &grads));
    for(int i = 0; i < variables.size(); i++)
    {
        train_ops.push_back(ApplyGradientDescent(scope, variables[i], learning_rate, grads[i]));
    }

    std::cout << "Starting episodes..." << std::endl;
    for(int numiter = start; numiter < params.nbiter; numiter++)
    {
        float decaying_rate = params.lr * exp((numiter / params.steplr) * log(params.gamma));
        std::vector<Tensor> tensor_list = generateInputsLabelsAndTarget(params, img_data);
        TF_CHECK_OK(session.Run({{net.inputs, tensor_list[0]}, {net.labels, tensor_list[1]}, {target, tensor_list[2]}, {learning_rate, decaying_rate}}, train_ops, NULL));
        TF_CHECK_OK(session.Run({{net.inputs, tensor_list[0]}, {net.labels, tensor_list[1]}, {target, tensor_list[2]}, {learning_rate, decaying_rate}}, {loss}, &outputs));
        FILE *file = fopen("loss_list.txt", "a");
        fprintf(file, "%d: %0.6f\n", numiter + 1, outputs[0].scalar<float>()(0));
        fclose(file);
        if((numiter + 1) % params.save_every == 0)
        {
            snapshot(numiter);
        }
    }
}

machine_learning::ResultType PlasticNetModel::evaluate(cv::Mat &test_image)
{
    const int nbsteps = params.nbshots * ((params.prestime + params.ipd) * params.nbclesses) + params.prestimetest;
    srand(params.rngseed);
    ImgData img_data(ml_datasets + "/train_datasets/miniimagenet/train/");
    this->restore( pretrained_model+ "/TensorFlow/pnn/plastic_net_1000000.ckpt");
    std::vector<Tensor> outputs = generateInputsLabelsAndTarget(params, img_data);
    cv::Mat resized_image;
    cv::resize(test_image, resized_image, cv::Size(params.ImgSize, params.ImgSize));
    for(int j = 0; j < params.ImgSize; j++)
    {
        for(int k = 0; k < params.ImgSize; k++)
        {
            outputs[0].tensor<float, 5>()(nbsteps - 1, 0, 0, j, k) = ((cv::Scalar*)(resized_image.data))[j * params.ImgSize + k](0);
        }
    }
    TF_CHECK_OK(session.Run({{net.inputs, outputs[0]}, {net.labels, outputs[1]}}, {net.out}, &outputs));
    auto y = outputs[0].vec<float>();
    std::cout << y << std::endl;
    return machine_learning::ResultType();
}

void PlasticNetModel::batch_evaluate(int size)
{
    srand(params.rngseed);
    ImgData img_data(ml_datasets + "/train_datasets/miniimagenet/train/");
    std::vector<int> indexList;
    int mistake = 0;
    this->restore(pretrained_model + "/TensorFlow/pnn/plastic_net_1000000.ckpt");
    std::cout << "Starting episodes..." << std::endl;
    for(int numiter = 0; numiter < size; numiter++)
    {
        std::vector<Tensor> tensor_list = generateInputsLabelsAndTarget(params, img_data);
        auto target_temp = tensor_list[2].vec<float>();

        std::vector<Tensor> outputs;
        TF_CHECK_OK(session.Run({{net.inputs, tensor_list[0]}, {net.labels, tensor_list[1]}}, {net.out}, &outputs));

        float max = 0;
        int index_target = 0;
        int index_y=0;
        for (int i=0;i<params.nbclesses;i++)
        {
            if(target_temp(i)>max)
            {
                max = target_temp(i);
                index_target = i;
            }
        }
        float max_y=0;

        auto y_temp = outputs[0].vec<float>();
        for(int j=0;j<params.nbclesses;j++)
        {
            std::cout<<"y_temp:"<<y_temp(j)<<std::endl;
            if(y_temp(j)>max_y)
            {
                max_y=y_temp(j);
                index_y = j;
                std::cout<<"max:"<<max_y<<std::endl;
            }
        }
        if(index_target != index_y)
        {
            mistake++;
        }
        indexList.push_back(index_y);
        std::cout<<"mistake:"<<mistake<<std::endl;
        std::cout<<"index_y:"<<index_y<<std::endl;
        std::cout<<"index_target:"<<index_target<<std::endl;
    }
    double accuracy = (double)mistake/(double)params.nbiter_test;
    std::cout<<"Mistake:"<<(double)(mistake/params.nbiter_test)<<std::endl;
    std::cout<<"Accuracy:"<<(1-accuracy)<<std::endl;
}
