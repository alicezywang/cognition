#pragma once
#ifndef __lowAltitudeSuppression_MODEL__H__
#define __lowAltitudeSuppression_MODEL__H__
#include <vector>
#include <math.h>
#include </usr/include/eigen3/Eigen/Dense>

#include <iostream>

namespace warning_expel_model
{


/**
* 低空掠过模型,输入为区域内人的位置、区域位置、边境位置，输出为最佳的入点位置和出点位置
*
*@points　   存放顺序为：人的经纬度　正方形四个框的经纬度(逆时针)　边境的经纬度
*       每一个点是一个size为２的std::vector<double>,[0]号元素为经度，[1]号元素为纬度
*@in_and_out 　入点和出点经纬度,顺序为：入点、出点、入点、出点
*@teams_wide  预留，编队宽
*@height    预留，编队高
*/
int lowAltitudeSuppression(const std::vector< std::vector<double> > & points , 
								 std::vector< std::vector<double> > & in_and_out,
								double teams_wide = 0,double height =0 );







}



#endif