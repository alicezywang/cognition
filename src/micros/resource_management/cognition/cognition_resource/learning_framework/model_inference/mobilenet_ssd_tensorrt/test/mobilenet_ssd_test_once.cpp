#include "mobilenet_ssd/mobilenet_ssd_tensorrt_model.h"
#include <opencv2/opencv.hpp>

using namespace machine_learning;

int main()
{
    //初始化model
    MobileNetSSDTensorRTModel TensorRT;
    //image测试
    //输入
    string package_name = "ml_datasets";
    string package_dir = ros::package::getPath(package_name);
    string image_path =  package_dir + "/val_datasets/test_pic/1 (1).jpg";
    cv::Mat cv_img = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);

    //认知资源调用测试
    ResultType results = TensorRT.evaluate(cv_img);
    //output
    vector<string> vec_classes = results.vec_classes;
    vector<float>  vec_scores  = results.vec_scores;
    vector<vector<float>> vec_bboxes = results.vec_bboxes;
    std::cout << "~~~~~~~vec_bboxes.size() = "<< vec_bboxes.size() << std::endl
              << vec_bboxes[0][0] << ", " << vec_bboxes[0][1] << std::endl
              << vec_bboxes[0][2] << ", " << vec_bboxes[0][3] << std::endl;

    //可视化
    for (int i = 0; i < vec_bboxes.size(); ++i) {
        int x = (int)vec_bboxes[i][0];
        int y = (int)vec_bboxes[i][1];
        int w = (int)vec_bboxes[i][2];
        int h = (int)vec_bboxes[i][3];
        cv::rectangle(cv_img, cv::Rect2d(x, y, w, h), cv::Scalar(255, 0, 0), 2, 1);
    }
    cv::imshow("Result", cv_img);
    cv::waitKey();
    cv::destroyWindow("Result");
    cout <<"~~~~~~~~可视化完成~~~~~~~~~~"<< endl;

    return 0;
}
