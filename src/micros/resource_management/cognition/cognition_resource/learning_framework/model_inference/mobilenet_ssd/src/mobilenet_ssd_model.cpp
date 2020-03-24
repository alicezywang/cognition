#include "mobilenet_ssd/mobilenet_ssd_model.h"

using namespace tensorflow;

namespace machine_learning
{

MobileNetSSDModel::MobileNetSSDModel()
{
    Status status = NewSession(SessionOptions(), &session);
    status = session->Create(network.graph_def);
    if(!status.ok())
    {
        throw std::runtime_error("Error creating session: " + status.ToString());
    }
}

MobileNetSSDModel::~MobileNetSSDModel()
{
    session->Close();
}

void MobileNetSSDModel::train(int start, int end)
{
    throw std::runtime_error("No train implementation.");
}

ResultType MobileNetSSDModel::evaluate(cv::Mat &test_image)
{
    //读取图像 test_image 到 input_image 中
    cv::Mat float_image, dts;
    test_image.convertTo(float_image, CV_32FC3);
    cv::resize(float_image, dts, cv::Size(300, 300));
    float *image_float_data = (float*)dts.data;
    tensorflow::TensorShape image_shape = tensorflow::TensorShape({1, dts.rows, dts.cols, dts.channels()});
    tensorflow::Tensor input_image = tensorflow::Tensor(tensorflow::DT_FLOAT, image_shape);
    std::copy_n(image_float_data, image_shape.num_elements(), input_image.flat<float>().data());

    std::vector<Tensor> outputs;
    const std::string input = "ToFloat:0";
    const std::string output1 = "detection_classes:0";
    const std::string output2 = "detection_scores:0";
    const std::string output3 = "detection_boxes:0";

    std::pair<std::string, Tensor> img(input, input_image);
    Status status = session->Run({img}, {output1, output2, output3}, {}, &outputs);
    if (!status.ok())
    {
        throw std::runtime_error("Running model failed: " + status.ToString());
    }

    const int mapping[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24, 25, 0, 0,
                           26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 0, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                           55, 56, 57, 58, 59, 0, 60, 0, 0, 61, 0, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 0, 73, 74, 75, 76, 77, 78, 79};
    ResultType result;
    for(int i = 0; i < outputs[0].dim_size(1); i++)
    {
        const float threshold = 0.5f;
        if(outputs[1].tensor<float,2>()(0,i) < threshold) continue;
        result.vec_classes.push_back(classNames[mapping[(int)(outputs[0].tensor<float,2>()(0,i))]]);
        result.vec_scores.push_back(outputs[1].tensor<float,2>()(0,i));
        result.vec_bboxes.push_back(std::vector<float>());
        result.vec_bboxes.back().push_back(outputs[2].tensor<float,3>()(0,i,1) * test_image.cols);
        result.vec_bboxes.back().push_back(outputs[2].tensor<float,3>()(0,i,0) * test_image.rows);
        result.vec_bboxes.back().push_back(outputs[2].tensor<float,3>()(0,i,3) * test_image.cols);
        result.vec_bboxes.back().push_back(outputs[2].tensor<float,3>()(0,i,2) * test_image.rows);
    }
    return result;
}

void MobileNetSSDModel::batch_evaluate(int size)
{
    throw std::runtime_error("No batch_evaluate implementation.");
}

}
