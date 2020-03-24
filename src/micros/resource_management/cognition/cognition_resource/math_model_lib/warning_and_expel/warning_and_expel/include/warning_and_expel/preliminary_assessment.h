#pragma once
#if !defined PRELIMINARY_ASSESSMENT_H
#define PRELIMINARY_ASSESSMENT_H

#include <iostream>
#include<math.h>
#include </usr/include/eigen3/Eigen/Dense>
#include<vector>


namespace warning_expel_model
{
/**
* 低空掠过预评估模型
*/


/**
* 群体低空掠过预评估模型
*@UAV_Breadth 威慑宽度
*@UAV_Breadth 威慑高度
*@UAV_number　无人机数量
*@in_and_out 每一架无人机的入点和出点，按照robot_1_入点，robot_1_出点,robot_２_入点,robot_２_出点输入
*@points　每个人的经纬度坐标<x,y>　＋＋　方框的四个点的坐标　＋＋　边境的两个点
*输出vector结构为：vector[0]为被威慑0次的人数;vector[1]为被威慑1次的人数;...最后一个数值vector[n]为威慑度
*/
std::vector <double> groupDeterrence(double UAV_Breadth,double UAV_Height,int UAV_number,const std::vector<std::vector<double> > in_and_out,const std::vector< std::vector<double> > & points);
//输出格式修改



/**
* 单体低空掠过预评估模型
*@UAV_Breadth 威慑宽度
*@UAV_Breadth 威慑高度
*@in_and_out 每一无人机的入点和出点，按照robot_1_入点经纬度，robot_1_出点经纬度输入
*@points　每个人的经纬度坐标<x,y>　＋＋　方框的四个点的坐标　＋＋　边境的两个点
*输出vector结构为：vector[i]为下标为i的人是否被威慑，0表示没有被威慑，１表示被威慑；最后一个数值vector[n]为威慑度
*/
std::vector <double> singleDeterrence(double UAV_Breadth,double UAV_Height,std::vector<std::vector<double> > _in_and_out,const std::vector< std::vector<double> > & points);





}
#endif
