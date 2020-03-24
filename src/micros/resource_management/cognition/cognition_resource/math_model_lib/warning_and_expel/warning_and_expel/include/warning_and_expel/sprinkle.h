#pragma once
#if !defined __SPRINKLE__MODEL_H_
#define __SPRINKLE__MODEL_H_

#include <iostream>
#include <vector>

namespace warning_expel_model
{

 /**
*喷洒模型半径判断
*＠h　无人机高度
*@angle　喷洒角度
*/
float sprinkle(float h,double angle = 80.0);

/**
*　喷洒直线队列,已知拐点，求无人机的分布点
* @interval                 两架无人机的间距
* @h                        无人机的高度
* @inflection_points        拐点，第二个纬度为经纬度
* @amount                   无人机的总量
* @leader                   leader的序号
* @airplane_points          返回值，返回的无人机分布的经度坐标。一组的长度：无人机的架数*2（上下／左右两个点）*2（经纬度）
* @r                        返回值，无人机的喷洒覆盖地面半径
* @y_direction              返回值，如果竖直方向走则返回true
* @angle                    无人机喷头的角度
* 
*/
int sprinkleLineInterval(float interval,float h,
                           std::vector<std::vector<double> > &inflection_points,
                           int amount,int leader,
                           std::vector<std::vector<double> > & airplane_points,
                           float & r,bool & y_direction,double angle = 80.0);
}
#endif
