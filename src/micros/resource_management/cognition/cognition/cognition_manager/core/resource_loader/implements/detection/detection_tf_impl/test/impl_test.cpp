#include "cognition_bus/detection_tf_impl.h"

using namespace std;

int main(int argc, char **argv)
{
    //根据任务的不同，初始化不同的认知总线
    cognition_bus::DetectionTFImpl cognition_softbus;

    //image测试
    //输入
    string package_name = "ml_datasets";
    string package_dir = ros::package::getPath(package_name);
    string image_path =  package_dir + "/val_datasets/test_pic/image5.jpg";
    cv::Mat cv_img = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
    vector<boost::any> inputs = {cv_img,};

    //认知资源调用测试
    vector<boost::any> results;
    cognition_softbus.call(inputs, results);

    //可视化
    cv::imshow("Result", cv_img);
    cv::waitKey();
    cv::destroyWindow("Result");
    cout <<"~~~~~~~~可视化完成~~~~~~~~~~"<< endl;

    //output
    vector<string> vec_classes = boost::any_cast<vector<string>>(results[0]);
    vector<float>  vec_scores  = boost::any_cast<vector<float>>(results[1]);
    vector<vector<float>> vec_bboxes = boost::any_cast<vector<vector<float>>>(results[2]);

    return 0;
}


