#ifndef FASTER_RCNN_MODEL_H
#define FASTER_RCNN_MODEL_H
// C and C++ headers
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/cc/client/client_session.h>

// ROS headers
#include <ros/package.h>

// Interface headers
#include <ml_model_base/model_base.h>

#include <vggnet_test.h>
#include <fast_rcnn/config.h>
#include <fast_rcnn/bbox_transform.h>
#include <utils/OpenCVPlot.h>
#include <nms/cpu_nms.h>

namespace machine_learning
{
using namespace tensorflow;
using namespace fast_rcnn;
using namespace std;

typedef Eigen::Tensor<float, 4, Eigen::RowMajor> Tensor4f;
typedef Eigen::Tensor<float, 3, Eigen::RowMajor> Tensor3f;
typedef Eigen::Tensor<float, 2, Eigen::RowMajor> Tensor2f;
typedef Eigen::Tensor<float, 1, Eigen::RowMajor> Tensor1f;
typedef Eigen::Tensor<int, 1, Eigen::RowMajor> Tensor1i;
typedef Eigen::DSizes<Eigen::Index, 2> Dimensions;
typedef unordered_map<Output, Input::Initializer, OutputHash> FeedType;

/**
 * @brief The FasterRCNNModel class
 */
class FasterRCNNModel : public ModelBase
{

public:
    FasterRCNNModel();
    ~FasterRCNNModel() = default;
    virtual void train(int start, int end);
    virtual ResultType evaluate(cv::Mat &cv_img);
    virtual void batch_evaluate(int size);

private:
    Scope scope;
    ClientSession session;
    boost::shared_ptr<vggnet_test> net; //VGG Net for test

    string pretrained_model;
    vector<string> CLASSES = {"__background__",
                             "aeroplane", "bicycle", "bird", "boat",
                             "bottle", "bus", "car", "cat", "chair",
                             "cow", "diningtable", "dog", "horse",
                             "motorbike", "person", "pottedplant",
                             "sheep", "sofa", "train", "tvmonitor"};
private:
    /**
     * @brief cvMat2tfTensor
     * @param cv_img
     * @param img_scale
     * @param image_tensor
     */
    void _cvMat2tfTensor(cv::Mat& cv_img, float& img_scale, tensorflow::Tensor& image_tensor);

    /**
     * @brief im_detect
     * @param sess
     * @param net
     * @param resized_im_it
     * @param im_scale
     * @param out_scores
     * @param out_pred_boxes
     * @return
     */
    bool _im_detect(ClientSession& sess, vggnet_test& net, const Tensor& resized_im_it, const float im_scale, Tensor2f& out_scores, Tensor2f& out_pred_boxes); //sess, net, im, boxes=None

    /**
     * @brief vis_detections
     * @param im
     * @param scores
     * @param boxes
     * @param nms_thresh
     * @param conf_thresh
     * @param vec_classes
     * @param vec_scores
     * @param vec_bboxes
     */
    void _vis_detections(cv::Mat &im, Tensor2f scores, Tensor2f boxes, float nms_thresh, float conf_thresh = 0.5,
                        vector<string> *vec_classes = nullptr, vector<float> *vec_scores = nullptr, vector<vector<float>> *vec_bboxes = nullptr);
    /**
     * @brief _clip_boxes
     * @param boxes
     * @param im_shape
     * @return
     */
    Tensor2f _clip_boxes(Tensor2f boxes,  Tensor1f im_shape);

};//class

}//namespace
#endif // FASTER_RCNN_MODEL_H
