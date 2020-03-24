#include "mobilenet_ssd/mobilenet_ssd_dnn_model.h"

namespace machine_learning
{

MobileNetSSDDNNModel::MobileNetSSDDNNModel()
{

}

MobileNetSSDDNNModel::~MobileNetSSDDNNModel()
{

}

void MobileNetSSDDNNModel::train(int start, int end)
{
    throw std::runtime_error("No train implementation.");
}

ResultType MobileNetSSDDNNModel::evaluate(cv::Mat &test_image)
{
    const int mapping[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24, 25, 0, 0,
                           26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 0, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                           55, 56, 57, 58, 59, 0, 60, 0, 0, 61, 0, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 0, 73, 74, 75, 76, 77, 78, 79};
    network.net->setInput(cv::dnn::blobFromImage(test_image, 1.0/255.0, Size(test_image.cols, test_image.rows)));
    Mat output = network.net->forward();
    Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());
    ResultType result;
    for(int i = 0; i < detectionMat.rows; i++)
    {
        const float threshold = 0.5f;
        if(detectionMat.at<float>(i, 2) < threshold) continue;
        result.vec_classes.push_back(classNames[mapping[(int)(detectionMat.at<float>(i, 1))]]);
        result.vec_scores.push_back(detectionMat.at<float>(i, 2));
        result.vec_bboxes.push_back(std::vector<float>());
        result.vec_bboxes.back().push_back(detectionMat.at<float>(i, 3) * test_image.cols);
        result.vec_bboxes.back().push_back(detectionMat.at<float>(i, 4) * test_image.rows);
        result.vec_bboxes.back().push_back(detectionMat.at<float>(i, 5) * test_image.cols);
        result.vec_bboxes.back().push_back(detectionMat.at<float>(i, 6) * test_image.rows);
    }
    return result;
}

void MobileNetSSDDNNModel::batch_evaluate(int size)
{
    throw std::runtime_error("No batch_evaluate implementation.");
}

}
