#include "cognition/cognition.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

int main(int argc, char **argv)
{
//TEST 1:
    //根据任务的不同，认知总线初始化不同的资源
    cognition::Handle cognition_handle_direct(cognition::UesType::Direct);

    //认知资源调用测试
    //get model path according model_id(<framework_name>_<model_name>_<dataset_name>)
    cognition::ModelMeta model = cognition_handle_direct.getModelFromId("TensorRT_MobileNetSSD_Car");
    cout << model.model_root_path << endl;

//TEST 2:
    cognition::task_description task;
    //paremeters from task_interface
    task.task_inf.task_id="001";    //WASM
    task.task_inf.task_type = "Detection";    
    task.task_inf.battle_field = "air";  
    task.task_inf.target_category = "car";  
    task.task_inf.target_status="static";
    task.task_inf.target_pose = 60;  //the z axis of target
    //paremeters from actor_interface
    task.actor_inf.actor_id="01";
    task.actor_inf.sensor_id="001";
    task.actor_inf.sensor_type = "visible";    
    task.actor_inf.cpu_info = "gpu";  
    task.actor_inf.memory_info = 0;  
    //根据任务的不同，认知总线初始化不同的资源
    cognition::Handle cognition_handle(cognition::UesType::Auto);
    cognition_handle.init(task);
    //cognition_handle.call();
    cout <<"~~~~~~~~完成~~~~~~~~~~"<< endl;

    return 0;
}
