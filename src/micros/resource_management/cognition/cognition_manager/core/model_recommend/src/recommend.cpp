#include <iostream>
#include <fstream>
#include <algorithm>
#include <queue>
#include <cmath>
#include <dirent.h>
#include <tinyxml2.h>
#include<string>
#include <stdio.h>
#include <stdlib.h>
#include <model_recommend/recommend.h>

namespace cognition {

using namespace tinyxml2;

static std::vector<model_description> get_model_list(std::string model_dir)
{
    std::vector<model_description> result;
    DIR *p_dir;
    dirent *p_item;
    p_dir = opendir(model_dir.c_str());
    //std::cout<<model_dir.c_str()<<std::endl;
    if(p_dir == NULL)
    { 
        std::cerr << "Loading directory " << model_dir << " failed." << std::endl;
        return result;
    }
    p_item = readdir(p_dir);
    int n=0;
    //数据写入csv文件
    FILE *fp = NULL; //需要注意
    std::string path=model_dir +"/"+"models_descriptions.csv";
    //fp = fopen("/media/hyl/C022AA4B225A6D42/test.csv", "w+");
    fp = fopen(path.c_str(), "w+");
    
    while(p_item != NULL)
    {
        //std::map<std::string, model_description>Model;
        const char *filename = p_item->d_name;
        int len = std::strlen(filename);
        if(len >= 4 && std::strcmp(filename + (len - 4), ".xml") == 0){
             
            XMLDocument *model_xml = new XMLDocument();
            XMLError error = model_xml->LoadFile((model_dir + "/" + filename).c_str());
            if (error != XML_SUCCESS){
                std::cerr << "Loading file " << filename << " failed." << std::endl;
                delete model_xml;
                closedir(p_dir);
                return result;
            }
            //访问根节点
            XMLElement *root = model_xml->RootElement();
            XMLElement* model_meta = root->FirstChildElement("model_meta");
            model_description temp;

            //访问model_meta的属性
            const XMLAttribute * model_id = model_meta->FirstAttribute();
            temp.model_id=model_id->Value();
            //遍历model_meta的其他子节点
            //----------------model------------------//
            XMLElement *train_framework = model_meta->FirstChildElement("train_framework");
            temp.train_framework = train_framework->GetText();
            XMLElement *confidence = model_meta->FirstChildElement("confidence");
            temp.confidence = atof(confidence->GetText());
            XMLElement *developer = model_meta->FirstChildElement("developer");
            temp.developer = developer->GetText();
            XMLElement *root_path = model_meta->FirstChildElement("root_path");
            temp.root_path = root_path->GetText();
            fprintf(fp, "%s,%s,%f,%s,%s,",temp.model_id.c_str(),temp.train_framework.c_str(),temp.confidence, temp.developer.c_str(),temp.root_path.c_str());
            //--------------------who---------------------//
            XMLElement *whoNodes = model_meta->FirstChildElement("who");
            XMLElement *sensor_type = whoNodes->FirstChildElement("sensor_type");
            temp.who_inf.sensor_type = sensor_type->GetText();

            XMLElement *memory_info = whoNodes->FirstChildElement("memory_info");
            temp.who_inf.memory_info = atoi(memory_info->GetText());

            XMLElement *cpu_info = whoNodes->FirstChildElement("cpu_info");
            temp.who_inf.cpu_info = cpu_info->GetText();
            fprintf(fp, "%s,%s,%d,",temp.who_inf.sensor_type.c_str(),temp.who_inf.cpu_info.c_str(), temp.who_inf.memory_info);
            //---------------------where----------------//
            XMLElement *whereNodes = model_meta->FirstChildElement("where");
            XMLElement *field = whereNodes->FirstChildElement("field");
            temp.where_inf.field = field->GetText();
            XMLElement *period = whereNodes->FirstChildElement("period");
            temp.where_inf.period = period->GetText();
            fprintf(fp, "%s,%s,",temp.where_inf.field.c_str(),temp.where_inf.period.c_str());
            //-----------------------what----------------//
            XMLElement *whatNodes = model_meta->FirstChildElement("what");
            XMLElement *task_type = whatNodes->FirstChildElement("task_type");
            temp.what_inf.task_type = task_type->GetText();
            XMLElement *target_category = whatNodes->FirstChildElement("target_category");
            temp.what_inf.target_category = target_category->GetText();
            XMLElement *target_status = whatNodes->FirstChildElement("target_status");
            temp.what_inf.target_status = target_status->GetText();
            XMLElement *target_pose = whatNodes->FirstChildElement("target_pose");
            temp.what_inf.target_pose = atof(target_pose->GetText());
            fprintf(fp, "%s,%s,%s,%f\n",temp.what_inf.task_type.c_str(),temp.what_inf.target_category.c_str(),temp.what_inf.target_status.c_str(),temp.what_inf.target_pose);

            result.insert(result.begin()+n,temp);
            delete model_xml;
            n=n+1;
	    }
        p_item = readdir(p_dir);
    }
    fclose(fp);
    closedir(p_dir);
    return result;
}

std::map<std::string, model_description> load_model_list()
{

    std::map<std::string, model_description> result;
    std::string package_path = ros::package::getPath("model_recommend");
    std::string model_data_path = package_path + "/model_descriptions";
    std::vector<model_description> model_list = get_model_list(model_data_path);//get_model_list
    //std::ofstream oss(model_data_path + "/tset_descriptions.csv");
    //std::cout<<"size = "<<model_list.size()<<std::endl;
    for(int i = 0; i < model_list.size(); i++){
        result[model_list[i].model_id]=model_list[i];
    }
    return result;
}

 static float normalize(const std::vector<int> &a,const std::vector<int> &b)
 {
     int len=a.size();
     float tem=0;
     for(int i=0;i<len;i++)
     {
         tem=tem+(a[i]-b[i])*(a[i]-b[i]);
     }
     tem=std::sqrt(tem);
 }

 static float distance(const task_description &task, const model_description &model)
{
    float result=0;
    int strlen;
    float xx,yy;
    std::vector<std::string> VecStr_task_type{"Detection","Classification","Track"};
    std::vector<std::string> VecStr_field{"air","land","sea"};
    std::vector<std::string> VecStr_period{"day","night"};
    std::vector<std::string> VecStr_target_category{"Car","Plane","Construction"};
    std::vector<std::string> VecStr_target_status{"dynamic","static"};
    std::vector<std::string> VecStr_sensor_type{"visible" ,"infrared","electromagnetic"};
    std::vector<std::string> VecStr_cpu_info{"cpu" ,"gpu"};
    //task_type
    strlen=VecStr_task_type.size();
    std::vector<int> a1(strlen,0);
    std::vector<int> b1(strlen,0);
    for (int i=0;i<strlen;i++){
        if (task.task_inf.task_type == VecStr_task_type[i]){
            a1[i]=1;
        }
        if (model.what_inf.task_type == VecStr_task_type[i]){
            b1[i]=1;
        }
    }
    result=result+normalize(a1,b1);
    //battle_filed
    strlen=VecStr_field.size();
    std::vector<int> a2(strlen,0);
    std::vector<int> b2(strlen,0);
    for (int i=0;i<strlen;i++){
        if (task.task_inf.battle_field == VecStr_field[i]){
            a2[i]=1;
        }
        if (model.where_inf.field == VecStr_field[i]){
            b2[i]=1;
        }
    }
    result=result+normalize(a2,b2);
    //task_period
    strlen=VecStr_period.size();
    std::vector<int> a3(strlen,0);
    std::vector<int> b3(strlen,0);
    for (int i=0;i<strlen;i++){
        if (task.task_inf.task_period == VecStr_period[i]){
            a3[i]=1;
        }
        if (model.where_inf.period == VecStr_period[i]){
            b3[i]=1;
        }
    }
    result=result+normalize(a3,b3);
    //target_category
    strlen=VecStr_target_category.size();
    std::vector<int> a4(strlen,0);
    std::vector<int> b4(strlen,0);
    for (int i=0;i<strlen;i++){
        if (task.task_inf.target_category == VecStr_target_category[i]){
            a4[i]=1;
        }
        if (model.what_inf.target_category == VecStr_target_category[i]){
            b4[i]=1;
        }
    }
    result=result+normalize(a4,b4);
    //target_status
    strlen=VecStr_target_status.size();
    std::vector<int> a5(strlen,0);
    std::vector<int> b5(strlen,0);
    for (int i=0;i<strlen;i++){
        if (task.task_inf.target_status == VecStr_target_status[i]){
            a5[i]=1;
        }
        if (model.what_inf.target_status == VecStr_target_status[i]){
            b5[i]=1;
        }
    }
    result=result+normalize(a5,b5);
    //target_pose
    xx=task.task_inf.target_pose;
    yy=model.what_inf.target_pose;
    result=result+std::abs(xx-yy)/std::max(xx,yy);
    //sensor_type
    strlen=VecStr_sensor_type.size();
    std::vector<int> a6(strlen,0);
    std::vector<int> b6(strlen,0);
    for (int i=0;i<strlen;i++){
        if (task.actor_inf.sensor_type == VecStr_sensor_type[i]){
            a6[i]=1;
        }
        if (model.who_inf.sensor_type == VecStr_sensor_type[i]){
            b6[i]=1;
        }
    }
    result=result+normalize(a6,b6);
    //cpu_info
    strlen=VecStr_cpu_info.size();
    std::vector<int> a7(strlen,0);
    std::vector<int> b7(strlen,0);
    for (int i=0;i<strlen;i++){
        if (task.actor_inf.cpu_info == VecStr_cpu_info[i]){
            a7[i]=1;
        }
        if (model.who_inf.cpu_info == VecStr_cpu_info[i]){
            b7[i]=1;
        }
    }
    result=result+normalize(a7,b7);
    //memory_info
    xx=task.actor_inf.memory_info;
    yy=model.who_inf.memory_info;
    if (std::max(xx,yy))
    result=result+std::abs(xx-yy)/std::max(xx,yy);
    return result;
}

std::vector<std::string> get_recommended_models(const task_description &task, const std::map<std::string, model_description> &model_list)
{
    std::vector<std::string> result_list;
    int model_length = model_list.size();
    //std::cout<<"mdoel_list size"<<model_length<<std::endl;
    std::map<std::string, model_description>::const_iterator iter;
    //get the result 初步删选出匹配的models
    for(iter = model_list.begin(); iter != model_list.end(); iter++)
    {
        if(task.task_inf.task_type == iter->second.what_inf.task_type)
        {
            if(task.actor_inf.sensor_type == iter->second.who_inf.sensor_type)
            {
                result_list.push_back(iter->first);
            }
        }
    }
    //sort the result
    int result_length = result_list.size();
    for(int i = 0; i < result_length; i++)
    {
        for(int j = 0; j < result_length - i - 1; j++)
        {   
            if(distance(task, model_list.at(result_list[j])) > distance(task, model_list.at(result_list[j+1])))
            {
                std::string temp = result_list[j];
                result_list[j] = result_list[j+1];
                result_list[j+1] = temp;
            }
            else{
                if(distance(task, model_list.at(result_list[j])) == distance(task, model_list.at(result_list[j+1])))
                {
                    if (model_list.at(result_list[j]).confidence < model_list.at(result_list[j+1]).confidence)
                    {
                        std::string temp = result_list[j];
                        result_list[j] = result_list[j+1];
                        result_list[j+1] = temp;
                    } 
                }
            }
        }
    }
    return result_list;
}

}//namespace
