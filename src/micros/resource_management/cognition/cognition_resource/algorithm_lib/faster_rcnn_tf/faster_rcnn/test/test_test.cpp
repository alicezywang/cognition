#include <fast_rcnn/config.h>
#include <fast_rcnn/test.h>
#include <utils/myTimer.h>
#include <roi_data_layer/roi_data_layer.h>
#include <bbox.h>

#include <iostream>

#include <tensorflow/core/lib/io/path.h>
#include <tensorflow/core/public/session.h>

// ROS interfasce
#include <ros/package.h>

using namespace fast_rcnn;
using namespace std;

int main(int argc, char **argv)
{
    //sleep(3);
    // 加载模型
    extern tensorflow::SessionOptions session_options;
    tensorflow::GPUOptions *g = new tensorflow::GPUOptions();
    g->set_allow_growth(true);
    session_options.config.set_allocated_gpu_options(g);
    Scope scope = Scope::NewRootScope();
    vggnet_test net(scope);
    ClientSession session(scope, session_options);
    cout <<"~~~~~~~~模型, 加载完毕~~~~~~~~~"<< endl;
    std::vector<std::string> weight_list = net.get_weight_list();
    std::vector<std::string> bias_list = net.get_bias_list();
    std::cout << net.get_output("bbox_pred").front().name() << std::endl;
    //std::vector<float> bbox_stds;
    //std::vector<float> bbox_means;
    //rdl::RoIDataLayer data_layer(rdl::get_data_layer(bbox_means, bbox_stds));
    for(int i=0; i<weight_list.size(); i++)
    {
        auto restored_tensor = Restore(scope, fast_rcnn::cfg.DATA_DIR + "pretrain_model/VGGnet_fast_rcnn_iter_70000.ckpt", weight_list[i] + "/weights", DT_FLOAT);
        TF_CHECK_OK(session.Run({Assign(scope, net.get_weight(weight_list[i]), restored_tensor)}, NULL));
        std::cout << weight_list[i] << std::endl;
    }
    for(int i=0; i<bias_list.size(); i++)
    {
        auto restored_tensor = Restore(scope, fast_rcnn::cfg.DATA_DIR + "pretrain_model/VGGnet_fast_rcnn_iter_70000.ckpt", bias_list[i] + "/biases", DT_FLOAT);
        TF_CHECK_OK(session.Run({Assign(scope, net.get_bias(bias_list[i]), restored_tensor)}, NULL));
        std::cout << bias_list[i] << std::endl;
    }
    cout <<"~~~~~模型-预训练参数,加载完毕~~~~~"<< endl;

    vector<string> image_set; //= {"000000561465.jpg", "000000561958.jpg", "000000562059.jpg", "000000562229.jpg", "000000562443.jpg", "000000562448.jpg", "000000562818.jpg", "000000562843.jpg", "000000563267.jpg", "000000563281.jpg", "000000563470.jpg", "000000563603.jpg", "000000563648.jpg", "000000563702.jpg", "000000563758.jpg", "000000563882.jpg", "000000564091.jpg", "000000564127.jpg", "000000564133.jpg", "000000564280.jpg", "000000564336.jpg", "000000565045.jpg", "000000565153.jpg", "000000565391.jpg", "000000565563.jpg", "000000565607.jpg", "000000565624.jpg", "000000565776.jpg", "000000565962.jpg", "000000566042.jpg", "000000566436.jpg", "000000567011.jpg", "000000567432.jpg", "000000568147.jpg", "000000568584.jpg", "000000568814.jpg", "000000568981.jpg", "000000569030.jpg", "000000569059.jpg", "000000569273.jpg", "000000569565.jpg", "000000569825.jpg", "000000569976.jpg", "000000570169.jpg", "000000570448.jpg", "000000570456.jpg", "000000570688.jpg", "000000570736.jpg", "000000570834.jpg", "000000571264.jpg", "000000571893.jpg", "000000572388.jpg", "000000572408.jpg", "000000572555.jpg", "000000572620.jpg", "000000572900.jpg", "000000572956.jpg", "000000573626.jpg", "000000573943.jpg", "000000574315.jpg", "000000574425.jpg", "000000574520.jpg", "000000574702.jpg", "000000574810.jpg", "000000575357.jpg", "000000575372.jpg", "000000575500.jpg", "000000575970.jpg", "000000576031.jpg", "000000576052.jpg", "000000576566.jpg", "000000576955.jpg", "000000577182.jpg", "000000577584.jpg", "000000577735.jpg", "000000577862.jpg", "000000577864.jpg", "000000577959.jpg", "000000577976.jpg", "000000578489.jpg", "000000578500.jpg", "000000578545.jpg", "000000578871.jpg", "000000578922.jpg", "000000579070.jpg", "000000579091.jpg", "000000579158.jpg", "000000579307.jpg", "000000579321.jpg", "000000579655.jpg", "000000579893.jpg", "000000579900.jpg", "000000579902.jpg", "000000580197.jpg", "000000580294.jpg", "000000580410.jpg", "000000580757.jpg", "000000581062.jpg", "000000581317.jpg", "000000581357.jpg"};
    vector<string> type_set;
    string ml_datasets = ros::package::getPath("ml_datasets");
    string datasets_path = ml_datasets + "/val_datasets/faster_rcnn_imagenet/";
    string lable_path    =  datasets_path + "/dataset.txt";
    cout << lable_path << endl;

    FILE *in_file = fopen(lable_path.c_str(), "r");
    char file_name[50], type[20];
    for(int i=0; i<2500; i++)
    {
        fscanf(in_file, "%s%s", file_name, type);
        image_set.push_back(string(file_name));
        type_set.push_back(string(type));
    }
    fclose(in_file);
        //FILE *file = fopen("test_timers.csv", "w");
        //for(int i=0; i<9; i++)
        //{
        //    fprintf(file, "%s,", ((std::string)net.timers[i].name()).c_str());
        //}
        //fprintf(file, "timer2,timer3,timer1,cvMat2tfTensor,total\n");
        //fclose(file);
    float count = 0.0f;
    //测试 data_layer.roidb_size/2 张图片
    //printf("All pictures: %d\n", data_layer.roidb_size>>1);
    for(int i=0; i<2500; i++)
    {
        float img_scale;
        //rdl::Blobs blob = data_layer.forward();
        tensorflow::Tensor image_tensor;
        //std::string image_path = tensorflow::io::JoinPath(cfg.DATA_DIR, "VOCdevkit/VOC2007/JPEGImages", data_layer.roidbs[i].image_name);
        cv::Mat cv_img = cv::imread( datasets_path + image_set[i], CV_LOAD_IMAGE_COLOR);
        Tensor2f scores, pred_boxes;
        float CONF_THRESH = 0.8; //按分值过滤
        float NMS_THRESH = 0.3;  //按IOU过滤: 非极大值抑制: 用于去除不必要的提议框
        myTimer timer, timer1;
          timer.tic();
            timer1.tic();
        cvMat2tfTensor(cv_img, img_scale, image_tensor);
            timer1.toc();
        im_detect(session, net, image_tensor, img_scale, scores, pred_boxes);
          timer.toc();
            //FILE *file = fopen("test_timers.csv", "a");
            //fprintf(file, "%0.6f,%0.6f\n", timer1.total_time, timer.total_time);
            //fclose(file);
        vector<string> vec_classes; vector<float> vec_scores; vector<vector<float> > vec_bboxes;
        vis_detections(cv_img, scores, pred_boxes, NMS_THRESH, CONF_THRESH, &vec_classes, &vec_scores, &vec_bboxes);
        /*int gt_len = blob.gt_boxes.dim_size(0);
        vector<vector<float> > gt_boxes;
        vector<int> gt_classes;
        for(int j=0; j<gt_len; j++)
        {
            vector<float> tmp;
            tmp.push_back(blob.gt_boxes.matrix<float>()(j,0) / img_scale);
            tmp.push_back(blob.gt_boxes.matrix<float>()(j,1) / img_scale);
            tmp.push_back(blob.gt_boxes.matrix<float>()(j,2) / img_scale);
            tmp.push_back(blob.gt_boxes.matrix<float>()(j,3) / img_scale);
            gt_boxes.push_back(tmp);
            gt_classes.push_back((int)blob.gt_boxes.matrix<float>()(j,4));
        }*/
        //int len = vec_classes.size(), tp = 0;
        /*vector<bool> gt_map(gt_len, false);
        vector<vector<float> > iou = overlaps(gt_boxes, vec_bboxes);
        for(int j=0; j<len; j++)
        {
            int l=-1;
            for(int k=0; k<gt_len; k++)
            {
                if(gt_map[k] == false && gt_classes[k] == vec_classes[j])
                {
                    if(l<0||iou[l][j]<iou[k][j])l=k;
                }
            }
            if(l >= 0 && iou[l][j] >= 0.6f)
            {
                tp++; gt_map[l]=true;
            }
        }*/
        //printf("Matching rate = %f\n", tp*2.0f/(len+gt_len)); count+=tp*2.0f/(len+gt_len);
        if(find(vec_classes.begin(), vec_classes.end(), type_set[i]) != vec_classes.end()) count+=1.0f;

        FILE *file = fopen("test_failed.txt", "a");
        if(find(vec_classes.begin(), vec_classes.end(), type_set[i]) == vec_classes.end()) fprintf(file, "%s\n", image_set[i].c_str());
        fclose(file);

        file = fopen("test_success.txt", "a");
        if(find(vec_classes.begin(), vec_classes.end(), type_set[i]) != vec_classes.end()) fprintf(file, "%s\n", image_set[i].c_str());
        fclose(file);

        file = fopen("accuracy.txt", "a");
        fprintf(file, "%0.0f %f\n", count, count * 100.0f / (i+1));
        fclose(file);
        //cv::imwrite(image_set[i], cv_img);

        printf("%f / %d = %f%%\n", count, i+1, count * 100.0f / (i+1));
        printf("~~~~~~~~~~~~~~~~~~~~~~~~iter %d cost time: %0.6f ms~~~~~~~~~~~~~~~~~~~~~~\n", i+1, timer.total_time);

    }
    return 0;
}
