#include <unistd.h>
#include <iostream>
#include <queue>
#include <tinyxml2.h>

#include <model_recommend/recommend.h>

using namespace tinyxml2;

int main()
{
    // api-1
    std::map<std::string, cognition::model_description> model_list = cognition::load_model_list();
    // task data
    std::string package_path = ros::package::getPath("model_recommend");
    XMLDocument *task_xml = new XMLDocument();
    XMLError error = task_xml->LoadFile((package_path + "/test/task_descriptions.xml").c_str());

    if (error != XML_SUCCESS)
    {
        std::cerr << "Loading file task_descriptions.xml failed." << std::endl;
        delete task_xml;
        return -1;
    }
    XMLElement *root = task_xml->RootElement();
    //XMLElement *task_meta = root->FirstChildElement("task_description");

    cognition::task_description task;
    //-------------actor_interface------------------//
    //XMLElement *actor_inf = task_meta->FirstChildElement("actor_interface");
    XMLElement *actor_inf = root->FirstChildElement("actor_interface");
    const XMLAttribute * actor_id = actor_inf->FirstAttribute();
    task.actor_inf.actor_id=actor_id->Value();
    //std::cout<<"id:"<<task.actor_inf.actor_id<<std::endl;
    XMLElement *sensor_type = actor_inf->FirstChildElement("sensor_type");
    task.actor_inf.sensor_type = sensor_type->GetText();
    XMLElement *memory_info = actor_inf->FirstChildElement("memory_info");
    task.actor_inf.memory_info = atoi(memory_info->GetText());
    XMLElement *cpu_info = actor_inf->FirstChildElement("cpu_info");
    task.actor_inf.cpu_info = cpu_info->GetText();
    //-------------task_interface------------------//
    //XMLElement *task_inf = task_meta->FirstChildElement("task_interface");
    XMLElement *task_inf = root->FirstChildElement("task_interface");
    const XMLAttribute * task_id = task_inf->FirstAttribute();
    task.task_inf.task_id=task_id->Value();
    XMLElement *task_type = task_inf->FirstChildElement("task_type");
    task.task_inf.task_type = task_type->GetText();
    XMLElement *battle_field = task_inf->FirstChildElement("battle_field");
    task.task_inf.battle_field = battle_field->GetText();
    XMLElement *task_period = task_inf->FirstChildElement("task_period");
    task.task_inf.task_period = task_period->GetText();
    XMLElement *target_category = task_inf->FirstChildElement("target_category");
    task.task_inf.target_category = target_category->GetText();
    XMLElement *target_status = task_inf->FirstChildElement("target_status");
    task.task_inf.target_status = target_status->GetText();
    XMLElement *target_pose = task_inf->FirstChildElement("target_pose");
    task.task_inf.target_pose = atof(target_pose->GetText());
    //recommend models
    std::vector<std::string> result_list = cognition::get_recommended_models(task, model_list);
    int result_length = result_list.size();
    std::cout<<"result_length = "<<result_length<<std::endl;
    //std::cout << task_id << ":";
    std::cout << task.task_inf.task_id << ":";
    for(int i = 0; i < result_length; i++)
    {
        std::cout << " " << result_list[i];
    }
    std::cout << std::endl;

    /*while(task_node != NULL)
    {
        std::string task_id = task_node->Attribute("task_id");
       
        
        // api-2-3
        std::vector<std::string> result_list = cognition::get_recommended_models(vectorize_task(task), model_list);
        int result_length = result_list.size();
        std::cout << task_id << ":";
        for(int i = 0; i < result_length; i++)
        {
            std::cout << " " << result_list[i];
        }
        std::cout << std::endl;
        task_node = task_node->NextSiblingElement();
    }*/
    delete task_xml;

    return 0;
}



