#ifndef TEST_H
#define TEST_H
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/cc/client/client_session.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vggnet_test.h>
#include <fast_rcnn/config.h>
#include <fast_rcnn/bbox_transform.h>
#include <utils/OpenCVPlot.h>
#include <nms/cpu_nms.h>

namespace fast_rcnn {
using namespace tensorflow;
using namespace std;

typedef Eigen::Tensor<float, 4, Eigen::RowMajor> Tensor4f;
typedef Eigen::Tensor<float, 3, Eigen::RowMajor> Tensor3f;
typedef Eigen::Tensor<float, 2, Eigen::RowMajor> Tensor2f;
typedef Eigen::Tensor<float, 1, Eigen::RowMajor> Tensor1f;
typedef Eigen::Tensor<int, 1, Eigen::RowMajor> Tensor1i;
typedef Eigen::DSizes<Eigen::Index, 2> Dimensions;
typedef unordered_map<Output, Input::Initializer, OutputHash> FeedType;

///函数作用: 核心功能之一,对外接口,需优先实现
bool im_detect(ClientSession& sess, vggnet_test& net, const Tensor& resized_im_it, const float im_scale, Tensor2f& out_scores, Tensor2f& out_pred_boxes); //sess, net, im, boxes=None
void cvMat2tfTensor(cv::Mat& cv_img, float& img_scale, tensorflow::Tensor& image_tensor);
void vis_detections(cv::Mat &im, Tensor2f scores, Tensor2f boxes, float nms_thresh, float conf_thresh = 0.5,
                        vector<string> *vec_classes = nullptr, vector<float> *vec_scores = nullptr, vector<vector<float>> *vec_bboxes = nullptr);

Tensor2f _clip_boxes(Tensor2f boxes,  Tensor1f im_shape);




}//namespace
#endif // TEST_H
