#include <fstream>
#include "fast_rcnn/test.h"
#include "utils/myTimer.h"

#define TIMER_DEBUG

namespace fast_rcnn{
string CLASSES[] = {"__background__",
           "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"};

//API : 读取图片
void cvMat2tfTensor(cv::Mat& cv_img, float& img_scale, tensorflow::Tensor& image_tensor){
    // minus means  //py-faster-rcnn中demo.py代码与C++版本的代码对比
                    //https://blog.csdn.net/u013553529/article/details/79029270?utm_source=copy
    cv::Mat cv_new(cv_img.rows, cv_img.cols, CV_32FC3, cv::Scalar(0,0,0));
    for (int h = 0; h < cv_img.rows; ++h ){
        for (int w = 0; w < cv_img.cols; ++w){
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[0])-float(102.9801);// Blue
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[1])-float(115.9465);// Green
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[2])-float(122.7717);// Red
        }
    }

    // resize: scale image to (1000, y) or (x, 600)
    const int  MAX_SIZE = 1000;
    const int  SCALE_SIZE = 600;
    int max_side = max(cv_img.rows, cv_img.cols);
    int min_side = min(cv_img.rows, cv_img.cols);
    cout << "(width, height) = (" << cv_img.cols << ", " << cv_img.rows << ")" << endl;
    img_scale = float(SCALE_SIZE) / float(min_side);
    if (round(float(max_side) * img_scale) > MAX_SIZE) {
        img_scale = float(MAX_SIZE) / float(max_side);
    }
    cout << "img_scale: " << img_scale << endl;

    int height = int(cv_img.rows * img_scale);
    int width = int(cv_img.cols * img_scale);
    cout << "re-scaled (width, height) = (" << width << ", " << height << ")" << endl;
    cv::Mat cv_resized;
    cv::resize(cv_new, cv_resized, cv::Size(width, height));

    //cv::Mat to tensorflow::Tensor
        // convert byte to float image
    cv::Mat image_float;
    cv_resized.convertTo(image_float, CV_32FC3);
    float *image_float_data = (float*)image_float.data;
        // create input shape
    tensorflow::TensorShape image_shape = tensorflow::TensorShape{1, cv_resized.rows, cv_resized.cols, cv_resized.channels()};
    image_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, image_shape);
    std::copy_n((char*) image_float_data, image_shape.num_elements() * sizeof(float),
                    const_cast<char*>(image_tensor.tensor_data().data()));
}

//API : 共用一个scope创建ClientSession运行对应的vggnet_test网络
bool im_detect(ClientSession& sess, vggnet_test& net, const Tensor& resized_im_it, const float im_scale, Tensor2f& out_scores, Tensor2f& out_pred_boxes){ //处理一张图片
    #ifdef TIMER_DEBUG
        //timer
        myTimer timer[3];
        timer[0].tic();
    #endif
    //feed data prepare
    int im_h = resized_im_it.shape().dim_size(1);
    int im_w = resized_im_it.shape().dim_size(2);
    Tensor im_info_it(DT_FLOAT,TensorShape({1,3}));
    im_info_it.tensor<float,2>().setValues({{float(im_h), float(im_w), im_scale}, });
    #ifdef TIMER_DEBUG
        //timer
        timer[0].toc();
        timer[1].tic();
    #endif
    //模型预测/推理
    FeedType inputs = {{net.data, resized_im_it}, {net.im_info, im_info_it}}; //输入
    vector<Tensor> outputs;                                                   //输出
    TF_CHECK_OK(sess.Run(inputs, {net.get_output("cls_score").front(),
                                  net.get_output("cls_prob").front(),
                                  net.get_output("bbox_pred").front(),
                                  net.get_output("rois").front(),
                                  net.timers[0],net.timers[1],net.timers[2],net.timers[3],net.timers[4],
                                  net.timers[5],net.timers[6],net.timers[7],net.timers[8]}, &outputs));
                //cout << "cls_prob  shape=" << outputs[1].shape().DebugString() << endl;
                //cout << "bbox_pred shape=" << outputs[2].shape().DebugString() << endl;
                //cout << "rois      shape=" << outputs[3].shape().DebugString() << endl;
    //cls_score, cls_prob, bbox_pred, rois = outputs[0:3]
    out_scores    = outputs[1].tensor<float,2>();
    Tensor2f rois = outputs[3].tensor<float,2>();                 //(n,5):n为box的个数;0列代表图片编号;1-4代表box的值
    Tensor2f box_deltas = outputs[2].tensor<float,2>();           //(n,84)
                //测试代码, 输出指定的运算节点输出
                //vector<Tensor> test_outputs;
                //TF_CHECK_OK(sess.Run(inputs, {net.get_output("conv5_3")}, &test_outputs));
                //int n = test_outputs[0].shape().num_elements(); ofstream ofs;
                //auto test_tensor_data = test_outputs[0].flat<float>().data();
                //ofs.open("test_tensor_conv5_3.txt");
                //for(int i=0; i<n; i++)
                //{
                //    ofs << fixed << setprecision(4) << ((float*)test_tensor_data)[i] << endl;
                //}
                //ofs.close();
    #ifdef TIMER_DEBUG
        //timer
        timer[1].toc();
        timer[2].tic();
    #endif
    //还原boxes的原大小
    Tensor2f boxes(rois.dimension(0), 4);
    boxes.chip(0,1) = rois.chip(1,1) / im_scale;
    boxes.chip(1,1) = rois.chip(2,1) / im_scale;
    boxes.chip(2,1) = rois.chip(3,1) / im_scale;
    boxes.chip(3,1) = rois.chip(4,1) / im_scale;

    if(cfg.TEST.BBOX_REG){
        Tensor1f im_shape(2);
        im_shape.setValues({im_h/im_scale, im_w/im_scale});        //im_shape为原始图片大小
        out_pred_boxes = bbox_transform_inv(boxes, box_deltas);
        out_pred_boxes = _clip_boxes(out_pred_boxes, im_shape);
    }
    else{
        //按列向量复制boxes(n,4)到
        //pred_boxes(n,84)
    }
    #ifdef TIMER_DEBUG
        //timer
        timer[2].toc();
        //save csv
        FILE *file = fopen("test_timers.csv", "a");
        for(int i=1; i<9; ++i){
            fprintf(file, ",%0.6f", outputs[i+4].scalar<double>()(0)-outputs[i+3].scalar<double>()(0));
        }
        fprintf(file, ",%0.6f,%0.6f,%0.6f,", timer[1].total_time, timer[2].total_time, timer[0].total_time);
        fclose(file);
    #endif
    return true; //scores; pred_boxes
}

//API : 识别结果可视化
void vis_detections(cv::Mat &im, Tensor2f scores, Tensor2f boxes, float nms_thresh, float conf_thresh,
                    vector<string> *vec_classes, vector<float> *vec_scores, vector<vector<float> > *vec_bboxes)
{
    int classify_num = boxes.dimension(1)/4;
    cout << im.size() << endl;
    cout << "classify_num = " << classify_num << endl;

    for(int i = 1; i < classify_num; ++i){
        Tensor2f dets(scores.dimension(0), 5);        //cls_boxes和cls_scores横向堆叠

        dets.chip(0, 1) = boxes.chip(4 * i, 1);
        dets.chip(1, 1) = boxes.chip(4 * i + 1, 1);
        dets.chip(2, 1) = boxes.chip(4 * i + 2, 1);
        dets.chip(3, 1) = boxes.chip(4 * i + 3, 1);
        dets.chip(4, 1) = scores.chip(i, 1);

        //第一次滤波
        Tensor1i keep = cpu_nms(dets, nms_thresh);
        //dets = dets[keep, :]
        int keep_dim0 = keep.dimension(0);
        Tensor2f dets_nms(keep_dim0, 5);               //(,0::3)=boxes;(,4)=scores
        for(int j = 0; j < keep_dim0; ++j)
        {
            dets_nms.chip(j, 0) = dets.chip(keep(j), 0);
        }

        //第二次滤波, 根据分值过滤box和score

        float score;
        vector<float> bbox;
        int dets_nms_dim0 = dets_nms.dimension(0);    //行数
        for(int j = 0; j < dets_nms_dim0; ++j){
            if(dets_nms(j, 4) >= conf_thresh){        //根据分值过滤后，得到剩余的提议窗口
                score = dets_nms(j, 4);               //取出score
                bbox.clear();
                bbox.push_back(dets_nms(j, 0));
                bbox.push_back(dets_nms(j, 1));
                bbox.push_back(dets_nms(j, 2));
                bbox.push_back(dets_nms(j, 3));

                //opencv可视化
                stringstream ss;
                ss << CLASSES[i] << ":" << setprecision(4) << score;
                drawRectOnImage(im, bbox);
                drawTextOnImage(im, bbox, ss.str());

                if(vec_classes != nullptr){
                    vec_bboxes->push_back(bbox);
                    vec_scores->push_back(score);
                    vec_classes->push_back(CLASSES[i]);
                }
            }
        }
    }
}

//tools : 修剪boxes到图像边沿
Tensor2f _clip_boxes(Tensor2f boxes,  Tensor1f im_shape){
//im_shape(0):hight; im_shape(1):weight
    Eigen::array<int, 1> dims({1});
    Tensor2f tmp(boxes.dimension(0), 2);

    for(int i = 0; i < boxes.dimension(1); i+=4){
        tmp.setZero();
        //1.x1 >= 0
        tmp.chip(0, 1) = boxes.chip(i, 1); //tem(num,2): 0列为boxes的值; 1列为0;
        boxes.chip(i, 1) = tmp.maximum(dims); //按列对比,取最大
        //2.y1 >= 0
        tmp.chip(0, 1) = boxes.chip(i+1, 1);
        boxes.chip(i+1, 1) = tmp.maximum(dims);

        tmp.chip(1, 1) = tmp.chip(1, 1) + float(im_shape(1) - 1); //将第1列改为im_shape[1] -1; 注意此处不支持隐式的类型转换
        //3.x2 < im_shape[1]
        tmp.chip(0, 1) = boxes.chip(i+2, 1);
        boxes.chip(i+2, 1) = tmp.minimum(dims);

        tmp.chip(1, 1).setZero();
        tmp.chip(1, 1) = tmp.chip(1, 1) + float(im_shape(0) - 1);
        //4.y2 < im_shape[0]
        tmp.chip(0, 1) = boxes.chip(i+3, 1);
        boxes.chip(i+3, 1) = tmp.minimum(dims);
    }
    return boxes;//(x1,y1,x2,y2)
}


} //namespace
