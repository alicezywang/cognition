#include "fast_rcnn/config.h"
#include <ros/package.h>
namespace fast_rcnn {
using namespace std;
using namespace boost;
using namespace boost::filesystem;

//! 调用此函数注意增加返回值非空判断, optional用于包装可能出现的非法值
optional<path> find_dir(const path& dir, const string& dirname ){
    typedef optional<path> result_type;
    if(!exists(dir) || !is_directory(dir)){
        return result_type();//返回空值
    }

    typedef recursive_directory_iterator  rd_iter; //文件迭代器
    rd_iter end;
    for(rd_iter pos(dir); pos != end; ++pos){
        if(is_directory(*pos) && pos->path().filename() == dirname){
          if(pos->path().parent_path().filename() != "lib")
            return result_type(pos->path().parent_path());
        }
    }

    return result_type();
}
//获取当前文件路径
size_t get_executable_path( char* processdir, size_t len)
{
    char* path_end;
    if(readlink("/proc/self/exe", processdir,len) <=0)
            return -1;
    path_end = strrchr(processdir,  '/');
    if(path_end == NULL)
            return -1;
    ++path_end;
    *path_end = '\0';
    return (size_t)(path_end - processdir);
}
//获取rootPath
string getRootPath()
{
    //新方法:获取一个ros包的绝对路径
    string package_name = "utils";
    string package_dir = ros::package::getPath(package_name);
    path package_path(package_dir);
        //cout <<"package_path: " << package_path.string() <<endl;
        //cout <<"root_path: " << package_path.parent_path().string() <<endl;
    return package_path.parent_path().string();
    //    //旧方法:获得包路径
    //    char mainPath[4960];
    //    get_executable_path(mainPath, sizeof(mainPath));
    //    path main_path(mainPath);
    //        //cout << "执行文件目录:"<< main_path.string() << endl;
    //            //string pathTail = "devel/lib/fast_rcnn/"; //新方法:直接按层级返回,确定包名
    //     path packagePath = main_path.parent_path().parent_path().parent_path().parent_path();
    //        //cout << "根目录:"<< packagePath << endl;

    //    auto dirpath = find_dir(packagePath /= "src", "rpn_msr");// /=会追加分隔符,+=只追加字符序列
    //    if(dirpath){
    //        string rootpath = (*dirpath).string();
    //        if(ros::ok()){
    //            //ros 参数服务器接口获取 rootpath
    //            ros::NodeHandle private_nh("~"); // 私有空间命名
    //            private_nh.param("rootPath", rootpath, rootpath);
    //            cout << "get 'rootPath' from rosparam is sucessed !"<< endl;
    //        }
    //        return rootpath;
    //    }
    //    else{
    //        return string(".");
    //    }
}

//对外数据接口
const CFG cfg;
string get_output_dir(string imdb_name, string weights_filename){
    //#e.g. Fsater-RCNN_TF/output/default/voc_2007_train
    path outdir(cfg.ROOT_DIR);
    outdir /= "output";
    outdir /= cfg.EXP_DIR;
    outdir /= imdb_name;
    if( !weights_filename.empty() ){
        outdir /= weights_filename;
    }
    if( !exists(outdir) ){
        create_directories(outdir);
        cout <<"create output_dir sucessed:" << outdir << endl;
    }
    return outdir.string();
}


} //namespace
