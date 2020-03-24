#include <fast_rcnn/config.h>
#include <fast_rcnn/test.h>
#include <utils/myTimer.h>

#include <iostream>
#include <thread>
#include <cstdlib>
#include <fstream>
#include <cassert>
#include <time.h>
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include <tensorflow/core/public/session.h>

using namespace fast_rcnn;
using namespace std;

void thread_test_image(string image_name)
{
//测试读取rois
    Tensor2f rois1(8, 4);
    Tensor1f scales1(4);
    rois1.setValues({{1.0,2.0,3.0,4.0},
                     {5.0,6.0,7.0,8.0},
                     {3.0,4.0,5.0,6.0},
                     {1.2,35.0,6.7,8.0},
                     {1.2,35.0,6.7,8.0},
                     {1.2,35.0,6.7,8.0},
                     {1.2,35.0,6.7,8.0},
                     {1.2,35.0,6.7,8.0}});
    scales1.setValues({3.0,1.0,4.0,2.0});
   //cout << "rois1 = " <<endl;
    //cout << rois1 << endl;
    //cout << "scales1 = " << scales1 << endl;
    Tensor2f test = _get_project_im_rois(rois1, scales1);
    //cout << "test = " << endl;
    //cout << test << endl;

    //测试读取image
    string image_path = tensorflow::io::JoinPath(cfg.DATA_DIR, "demo", image_name);
    float im_scale;
    vector<Tensor> resized_tensors;
    TF_CHECK_OK(_ReadTensorFromImageFile(image_path, im_scale, &resized_tensors));
    const Tensor& resized_tensor = resized_tensors[0];
//        cout << "读取图片成功:" << endl;
//        cout << "图片形状:" << resized_tensor.shape().DebugString() <<endl;
//        cout << "图片维度:" << resized_tensor.shape().dims() <<endl;
//        cout << "图片高度:" << resized_tensor.shape().dim_size(1) <<endl;
//        cout << "图片宽度:" << resized_tensor.shape().dim_size(2) <<endl;
//        cout << "图片尺度:" << im_scale <<endl;
    // 加载模型
    Scope scope = Scope::NewRootScope();
    vggnet_test net(scope);
    ClientSession session(scope);

    cout <<"~~~~~~~~模型, 加载完毕~~~~~~~~~"<< endl;

    std::vector<Tensor> outputs;
    std::vector<std::string> weight_list = net.get_weight_list();
    std::vector<std::string> bias_list = net.get_bias_list();

    for(int i=0; i<weight_list.size(); i++)
    {
        auto restored_tensor = Restore(scope, fast_rcnn::cfg.DATA_DIR + "VGGnet_fast_rcnn_iter_70000.ckpt", weight_list[i] + "/weights", DT_FLOAT);
        TF_CHECK_OK(session.Run({Assign(scope, net.get_weight(weight_list[i]), restored_tensor)}, NULL));
        std::cout << weight_list[i] << std::endl;
    }
    for(int i=0; i<bias_list.size(); i++)
    {
        auto restored_tensor = Restore(scope, fast_rcnn::cfg.DATA_DIR + "VGGnet_fast_rcnn_iter_70000.ckpt", bias_list[i] + "/biases", DT_FLOAT);
        TF_CHECK_OK(session.Run({Assign(scope, net.get_bias(bias_list[i]), restored_tensor)}, NULL));
        std::cout << bias_list[i] << std::endl;
    }

    cout <<"~~~~~模型-预训练参数,加载完毕~~~~~"<< endl;

    // 获取图像识别结果
    Tensor2f scores;
    Tensor2f pred_boxes;
    myTimer timer;
    timer.tic();
    im_detect(session, net, image_path, scores, pred_boxes);

    timer.toc();

    cout << "Detection took " << timer.total_time << "ms for "
         << pred_boxes.dimension(0) <<" object proposals"<< endl;
    cout <<"~~~~~~~~图像识别完成~~~~~~~~~~~"<< endl;

    float CONF_THRESH = 0.8; //按分值过滤
    float NMS_THRESH = 0.3;  //按IOU过滤:非极大值抑制:用于去除不必要的提议boxe
    vis_detections(image_path, scores, pred_boxes, NMS_THRESH, CONF_THRESH);
    cout <<"~~~~~~~~数据可视化完成~~~~~~~~~~"<< endl;
}

int main(int argc, char **argv)
{

    std::system("ls ./src/faster_rcnn_tf/data/demo >> ./src/faster_rcnn_tf/data/imagelist.txt");

    ifstream infile;
    infile.open("./src/faster_rcnn_tf/data/imagelist.txt");

    string image_name_temp;
    vector<string> image_list;
    while(getline(infile,image_name_temp))
    {
        image_list.push_back(image_name_temp);
        cout<<image_name_temp<<endl;
    }

    std::system("rm ./src/faster_rcnn_tf/data/imagelist.txt");

    for(int i=0;i<image_list.size();i++)
    {
        thread t(thread_test_image,image_list[i]);

        if(i+1!=image_list.size())
        {
        thread x(thread_test_image,image_list[i+1]);
        x.join();
        }
        t.join();
        i++;
    }
}

