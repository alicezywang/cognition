#include "fast_rcnn/config.h"
#include <iostream>
//#include <sys/stat.h> //create_directories(p1)
//#include <sys/types.h>

using namespace fast_rcnn;
using namespace std;

int main(int argc, char **argv)
{
//    ros::init(argc, argv, "myclass");
    path p1("./11/output");
    //create_directories(p1); //创建多级目录

    cout << "file/dir path : "<< p1.string() << endl;
    cout << "current_path : "<< p1.native() << endl;
    cout << "current_path : "<< current_path() << endl;
    cout << "initial_path : "<< initial_path() << endl;

    //string rootPath = getRootPath();
    //cout <<"search faster_rcnn_tf path from getRootPath() sucessed :"<< rootPath <<endl;

    //string outPath = get_output_dir("11");
    //cout <<"search outPath from get_output_dir() sucessed :"<< outPath <<endl;

    cout << cfg.TRAIN.LEARNING_RATE << endl;
    cout << cfg.USE_GPU_NMS << endl;
    cout << cfg.ROOT_DIR <<endl;
    cout << cfg.DATA_DIR <<endl;
    return 0;
}

