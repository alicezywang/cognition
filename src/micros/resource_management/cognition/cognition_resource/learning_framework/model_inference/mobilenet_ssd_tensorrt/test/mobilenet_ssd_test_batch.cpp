#include "mobilenet_ssd/mobilenet_ssd_tensorrt_model.h"

using namespace machine_learning;
using namespace std;
class Timer2 {
public:
    void tic()
    {
        start_ticking_ = true;
        start_ = std::chrono::high_resolution_clock::now();
    }
    void toc()
    {
        if (!start_ticking_)return;
        end_ = std::chrono::high_resolution_clock::now();
        start_ticking_ = false;
        t = std::chrono::duration<double, std::milli>(end_ - start_).count();
        //ROS_INFO("Time: %f ms", t);
    }
    double t;
private:
    bool start_ticking_ = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_;
};

const int MaxNum=3;

int main(int argc, char **argv)
{
    //初始化model
    MobileNetSSDTensorRTModel TensorRT;
    //image测试
    //输入
    string package_name = "ml_datasets";
    string package_dir = ros::package::getPath(package_name);
    char filename[100];
    double sumMs = 0;
    int count = 0;
    for(int j=0;j<100;j++)
    {
        for (int i = 1; i < MaxNum; i++)
        {
            sprintf(filename, (package_dir + "/val_datasets/test_pic/1 (%d).jpg").c_str(), i);
            // 读入图片
            cv::Mat cv_img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
            if (!cv_img.data)
            {
              cout<<"Picture loading failed"<<endl;
              break;
            }

            Timer2 ti;
            ti.tic();
            count++;
            //认知资源调用测试
            ResultType results = TensorRT.evaluate(cv_img);
            ti.toc();

            double reTime=ti.t;
            sumMs+=reTime;
            cout<<"Calling Per Times: "<<reTime<<" ms "<<endl;
            vector<string> vec_classes = results.vec_classes;
            vector<float>  vec_scores  = results.vec_scores;
            vector<vector<float>> vec_bboxes = results.vec_bboxes;
            std::cout << "~~~~~~~vec_bboxes.size() = "<< vec_bboxes.size() << std::endl;

            //可视化
            for (int i = 0; i < vec_bboxes.size(); ++i) {
              int x = (int)vec_bboxes[i][0]-15;
              int y = (int)vec_bboxes[i][1]-8;
              int w = (int)vec_bboxes[i][2];
              int h = (int)vec_bboxes[i][3];
              cv::rectangle(cv_img, cv::Rect2d(x, y, w, h), cv::Scalar(255, 0, 0), 2, 1);
            }

            cv::Mat dst = cv::Mat::zeros(cv_img.rows/3, cv_img.cols/3, CV_8UC3);
            resize(cv_img, dst, dst.size());
            cv::imshow("Result", dst);
            cv::waitKey(2);
            cout <<"~~~~~~~~可视化完成~~~~~~~~~~"<< endl;
        }
    }

    cout<<"总时间 = "<<sumMs<<" ms "<<" "<<"总次数 = "<<count<<endl;
    cout<<"Average Time : "<<sumMs/count<<" ms "<<" 帧率: "<<1000.000/(sumMs/count)<<" fps "<<endl;
    return 0;
}
