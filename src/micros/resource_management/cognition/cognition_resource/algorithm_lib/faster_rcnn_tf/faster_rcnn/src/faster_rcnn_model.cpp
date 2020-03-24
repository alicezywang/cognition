#include "fast_rcnn/faster_rcnn_model.h"
#include <utils/myTimer.h>

using namespace machine_learning;

FasterRCNNModel::FasterRCNNModel()
    :scope(Scope::NewRootScope()),
     session(scope)
{
    pretrained_model = ros::package::getPath("pretrained_model");
    string pretrained_model = pretrained_model + "/TensorFlow/faster_rcnn/" + "VGGnet_fast_rcnn_iter_70000.ckpt";

    // 加载模型
    net = boost::make_shared<vggnet_test>(scope);

    // model restore:
    cout <<"~~~ RCNN Model Restore ... ~~~"<< endl;
    std::vector<std::string> weight_list = net->get_weight_list();
    std::vector<std::string> bias_list = net->get_bias_list();
    std::cout << net->get_output("bbox_pred").front().name() << std::endl;
    for(int i=0; i<weight_list.size(); i++)
    {
        auto restored_tensor = Restore(scope, pretrained_model, weight_list[i] + "/weights", DT_FLOAT);
        TF_CHECK_OK(session.Run({Assign(scope, net->get_weight(weight_list[i]), restored_tensor)}, NULL));
        std::cout << weight_list[i] << std::endl;
    }
    for(int i=0; i<bias_list.size(); i++)
    {
        auto restored_tensor = Restore(scope, pretrained_model, bias_list[i] + "/biases", DT_FLOAT);
        TF_CHECK_OK(session.Run({Assign(scope, net->get_bias(bias_list[i]), restored_tensor)}, NULL));
        std::cout << bias_list[i] << std::endl;
    }
}

ResultType FasterRCNNModel::evaluate(cv::Mat &cv_img) {
    //output
    ResultType result;
    auto vec_bboxes_  = &result.vec_bboxes;
    auto vec_scores_  = &result.vec_scores;
    auto vec_classes_ = &result.vec_classes;

    //输入数据预处理
    float img_scale;
    tensorflow::Tensor image_tensor;
    _cvMat2tfTensor(cv_img, img_scale, image_tensor);

    //中间运算
    myTimer timer;
    timer.tic();

    Tensor2f scores;
    Tensor2f pred_boxes;
    _im_detect(session, *net, image_tensor, img_scale, scores, pred_boxes);

    float CONF_THRESH = 0.8f; //按分值过滤
    float NMS_THRESH = 0.3f;  //按IOU过滤:非极大值抑制:用于去除不必要的提议boxe
    _vis_detections(cv_img, scores, pred_boxes, NMS_THRESH, CONF_THRESH, vec_classes_, vec_scores_, vec_bboxes_);

    timer.toc();
    printf("cost time = %0.6f ms\n", timer.total_time);

    //返回值
    return result;
}

void FasterRCNNModel::batch_evaluate(int size) {
    //val datasets
    string ml_datasets = ros::package::getPath("ml_datasets");
    string datasets_path = ml_datasets + "/val_datasets/faster_rcnn_imagenet/";
    string lable_path    =  datasets_path + "/dataset.txt";

    vector<string> image_set; //= {"000000561465.jpg", "000000561958.jpg", "000000562059.jpg", "000000562229.jpg", "000000562443.jpg", "000000562448.jpg", "000000562818.jpg", "000000562843.jpg", "000000563267.jpg", "000000563281.jpg", "000000563470.jpg", "000000563603.jpg", "000000563648.jpg", "000000563702.jpg", "000000563758.jpg", "000000563882.jpg", "000000564091.jpg", "000000564127.jpg", "000000564133.jpg", "000000564280.jpg", "000000564336.jpg", "000000565045.jpg", "000000565153.jpg", "000000565391.jpg", "000000565563.jpg", "000000565607.jpg", "000000565624.jpg", "000000565776.jpg", "000000565962.jpg", "000000566042.jpg", "000000566436.jpg", "000000567011.jpg", "000000567432.jpg", "000000568147.jpg", "000000568584.jpg", "000000568814.jpg", "000000568981.jpg", "000000569030.jpg", "000000569059.jpg", "000000569273.jpg", "000000569565.jpg", "000000569825.jpg", "000000569976.jpg", "000000570169.jpg", "000000570448.jpg", "000000570456.jpg", "000000570688.jpg", "000000570736.jpg", "000000570834.jpg", "000000571264.jpg", "000000571893.jpg", "000000572388.jpg", "000000572408.jpg", "000000572555.jpg", "000000572620.jpg", "000000572900.jpg", "000000572956.jpg", "000000573626.jpg", "000000573943.jpg", "000000574315.jpg", "000000574425.jpg", "000000574520.jpg", "000000574702.jpg", "000000574810.jpg", "000000575357.jpg", "000000575372.jpg", "000000575500.jpg", "000000575970.jpg", "000000576031.jpg", "000000576052.jpg", "000000576566.jpg", "000000576955.jpg", "000000577182.jpg", "000000577584.jpg", "000000577735.jpg", "000000577862.jpg", "000000577864.jpg", "000000577959.jpg", "000000577976.jpg", "000000578489.jpg", "000000578500.jpg", "000000578545.jpg", "000000578871.jpg", "000000578922.jpg", "000000579070.jpg", "000000579091.jpg", "000000579158.jpg", "000000579307.jpg", "000000579321.jpg", "000000579655.jpg", "000000579893.jpg", "000000579900.jpg", "000000579902.jpg", "000000580197.jpg", "000000580294.jpg", "000000580410.jpg", "000000580757.jpg", "000000581062.jpg", "000000581317.jpg", "000000581357.jpg"};
    vector<string> type_set;

    FILE *in_file = fopen(lable_path.c_str(), "r");
    char file_name[50], type[20];
    for(int i=0; i<2500; i++)
    {
        fscanf(in_file, "%s%s", file_name, type);
        image_set.push_back(string(file_name));
        type_set.push_back(string(type));
    }
    fclose(in_file);

    //batch evaluate
    float count = 0.0f;
    for(int i=0; i<2500; i++)
    {
        //1. image input
        float img_scale;
        tensorflow::Tensor image_tensor;
        cv::Mat cv_img = cv::imread( datasets_path + image_set[i], CV_LOAD_IMAGE_COLOR);
        Tensor2f scores, pred_boxes;

        //2. model evaluate
        float CONF_THRESH = 0.8; //按分值过滤
        float NMS_THRESH = 0.3;  //按IOU过滤: 非极大值抑制: 用于去除不必要的提议框
        vector<string> vec_classes; vector<float> vec_scores; vector<vector<float> > vec_bboxes;

        myTimer timer, timer1;
        timer.tic();
            timer1.tic();
        _cvMat2tfTensor(cv_img, img_scale, image_tensor);
            timer1.toc();
        _im_detect(session, *net, image_tensor, img_scale, scores, pred_boxes);
        timer.toc();
        _vis_detections(cv_img, scores, pred_boxes, NMS_THRESH, CONF_THRESH, &vec_classes, &vec_scores, &vec_bboxes);

        //3. save result
        if(find(vec_classes.begin(), vec_classes.end(), type_set[i]) != vec_classes.end())
            count+=1.0f; //right evaluate count

        FILE *file = fopen("test_failed.txt", "a");
        if(find(vec_classes.begin(), vec_classes.end(), type_set[i]) == vec_classes.end()) fprintf(file, "%s\n", image_set[i].c_str());
        fclose(file);

        file = fopen("test_success.txt", "a");
        if(find(vec_classes.begin(), vec_classes.end(), type_set[i]) != vec_classes.end()) fprintf(file, "%s\n", image_set[i].c_str());
        fclose(file);

        file = fopen("accuracy.txt", "a");
        fprintf(file, "%0.0f %f\n", count, count * 100.0f / (i+1));
        fclose(file);
        //cv::imwrite(image_set[i], cv_img); //save evaluated image in current path

        printf("%f / %d = %f%%\n", count, i+1, count * 100.0f / (i+1));
        printf("~~~~~~~~~~~~~~~~~~~~~~~~iter %d cost time: %0.6f ms~~~~~~~~~~~~~~~~~~~~~~\n", i+1, timer.total_time);

        file = fopen("batch_evaluate_time.txt", "a");
        fprintf(file, "%f\n", timer.total_time);
        fclose(file);
    }

}

void FasterRCNNModel::train(int start, int end) {

}


//internal API : 读取图片
void FasterRCNNModel::_cvMat2tfTensor(cv::Mat& cv_img, float& img_scale, tensorflow::Tensor& image_tensor){
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

//internal API : 创建ClientSession, 运行对应的vggnet_test网络
bool FasterRCNNModel::_im_detect(ClientSession& sess, vggnet_test& net, const Tensor& resized_im_it, const float im_scale, Tensor2f& out_scores, Tensor2f& out_pred_boxes){ //处理一张图片
    //feed data prepare
    int im_h = resized_im_it.shape().dim_size(1);
    int im_w = resized_im_it.shape().dim_size(2);
    Tensor im_info_it(DT_FLOAT,TensorShape({1,3}));
    im_info_it.tensor<float,2>().setValues({{float(im_h), float(im_w), im_scale}, });

    //模型预测/推理
    FeedType inputs = {{net.data, resized_im_it}, {net.im_info, im_info_it}}; //输入
    vector<Tensor> outputs;                                                   //输出
    TF_CHECK_OK(sess.Run(inputs, {net.get_output("cls_score").front(),
                                  net.get_output("cls_prob").front(),
                                  net.get_output("bbox_pred").front(),
                                  net.get_output("rois").front(),
                                  net.timers[0],net.timers[1],net.timers[2],net.timers[3],net.timers[4],
                                  net.timers[5],net.timers[6],net.timers[7],net.timers[8]}, &outputs));
    //cls_score, cls_prob, bbox_pred, rois = outputs[0:3]
    out_scores    = outputs[1].tensor<float,2>();
    Tensor2f rois = outputs[3].tensor<float,2>();                 //(n,5):n为box的个数;0列代表图片编号;1-4代表box的值
    Tensor2f box_deltas = outputs[2].tensor<float,2>();           //(n,84)
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
    return true; //scores; pred_boxes
}

//internal API : 识别结果可视化
void FasterRCNNModel::_vis_detections(cv::Mat &im, Tensor2f scores, Tensor2f boxes, float nms_thresh, float conf_thresh,
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

//internal API : 修剪boxes到图像边沿
Tensor2f FasterRCNNModel::_clip_boxes(Tensor2f boxes,  Tensor1f im_shape){
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
