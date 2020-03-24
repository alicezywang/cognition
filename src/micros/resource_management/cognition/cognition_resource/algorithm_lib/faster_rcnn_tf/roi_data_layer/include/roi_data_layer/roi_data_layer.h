#ifndef _ROIDATALAYER_H
#define _ROIDATALAYER_H

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include <unsupported/Eigen/CXX11/Tensor>

#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/cc/client/client_session.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fast_rcnn/config.h>
#include <datasets/pascal_voc.h>

namespace rdl{
using namespace std;
using namespace fast_rcnn;
using namespace tensorflow;

typedef Eigen::Tensor<float, 4, Eigen::RowMajor> Tensor4f;
typedef Eigen::Tensor<float, 3, Eigen::RowMajor> Tensor3f;
typedef Eigen::Tensor<float, 2, Eigen::RowMajor> Tensor2f;
typedef Eigen::Tensor<float, 1, Eigen::RowMajor> Tensor1f;
typedef Eigen::Tensor<int, 1> Tensor1i;
typedef Eigen::DSizes<Eigen::Index, 2> Dimensions;


struct Blobs
{
    Tensor data;
    Tensor gt_boxes;
    Tensor im_info;
};


class RoIDataLayer{
public:
    RoIDataLayer(std::vector<annotation_prepare> get_roidbs);
    ~RoIDataLayer();
    // Tensor1i minibatch(cfg.TRAIN.IMS_PER_BATCH);
    Blobs forward();

private:
    int roidb_size;
    int num_classes;
    int cur;            //current inds of epoch
    int IMS_PER_BATCH;  //from cfg
    std::vector<annotation_prepare> roidbs;
    std::vector<annotation_prepare> minibatch_db;
    Tensor1i perm;      //用于存放shuffle_roidb_inds生成的随机数组
    
    void _shuffle_roidb_inds();
    Blobs _get_next_minibatch();
        Tensor1i _get_next_minibatch_inds();
        Blobs _get_minibatch(vector<annotation_prepare> batch_roidbs);

    //读取图片
    void _get_image_blob(string& im_path, Tensor& image, float& scale);
        void cvMat2tfTensor(cv::Mat& cv_img, float& img_scale, tensorflow::Tensor& image_tensor);
        bool EndsWith(const string &a, const string &b);
        Status _ReadEntireFile(tensorflow::Env* env, const string& filename, Tensor* output);
        Status _ReadTensorFromImageFile(const string &file_name, float &im_scale, std::vector<Tensor> *out_tensors);
};


RoIDataLayer get_data_layer(vector<float> &means_ravel, vector<float> &stds_ravel);


}//namespace

#endif