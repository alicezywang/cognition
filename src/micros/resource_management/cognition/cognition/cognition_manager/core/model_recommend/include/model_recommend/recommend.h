#ifndef RECOMMEND_H
#define RECOMMEND_H

#include <vector>
#include <string>
#include <map>
#include <ros/package.h>

namespace cognition {

struct task_info
{
    std::string task_id;
    std::string task_type;
    std::string battle_field;
    std::string task_period;
    std::string target_category;
    std::string target_status;
    float target_pose;
};

struct actor_info
{
    std::string actor_id;
    std::string sensor_id;
    std::string sensor_type;
    std::string cpu_info;
    int memory_info;
};

struct task_description
{
    task_info task_inf;
    actor_info actor_inf;
};

struct actor_who
{
    std::string sensor_type;
    std::string cpu_info;
    int memory_info;
};

struct task_what
{
    std::string task_type;
    std::string target_category;
    std::string target_status;
    float target_pose;
};

struct scene_where
{
    std::string field;
    std::string period;
};


struct model_description
{
    std::string model_id;
    std::string train_framework;
    std::string developer;
    std::string root_path;
    float confidence;
    actor_who who_inf;
    task_what what_inf;
    scene_where where_inf;
};


std::map<std::string, model_description> load_model_list();
std::vector<std::string> get_recommended_models(const task_description &task, const std::map<std::string, model_description> &model_list);

}//namespace
#endif // RECOMMEND_H_H
