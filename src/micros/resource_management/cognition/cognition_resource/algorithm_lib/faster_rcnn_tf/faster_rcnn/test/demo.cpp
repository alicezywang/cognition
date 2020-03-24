#include <fast_rcnn/faster_rcnn_model.h>
#include <fast_rcnn/config.h>

using namespace machine_learning;

int main(int argc, char **argv)
{
    sleep(3);
    FasterRCNNModel faster_rcnn_model;

    string ml_datasets = ros::package::getPath("ml_datasets");
    string image_path = ml_datasets + "/val_datasets/faster_rcnn_coco/" + "004545.jpg";
    cv::Mat cv_img = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);

    faster_rcnn_model.evaluate(cv_img);

    //可视化
    cv::imshow("Result", cv_img);
    cv::waitKey();
    cv::destroyWindow("Result");
    cout <<"~~~~~~~~可视化完成~~~~~~~~~~"<< endl;

    //batch evaluate
    faster_rcnn_model.batch_evaluate(2500);
    cout <<"~~~~~~~~~batch_evaluate completed~~~~~~~~~"<< endl;

    return 0;
}
