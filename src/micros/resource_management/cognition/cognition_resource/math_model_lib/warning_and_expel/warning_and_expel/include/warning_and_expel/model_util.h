#pragma once
#ifndef __MODEL_UTIL__H__
#define __MODEL_UTIL__H__
#include <vector>
#include <math.h>
#include </usr/include/eigen3/Eigen/Dense>
#include <iostream>

namespace warning_expel_model
{




/**
* 坐标转换为东北天的平面坐标
*
*@longitude  　　  目标经度
*@latitude 　　　　　　目标纬度
*@opoint_long  　　原点经度
*@opoint_lat   原点纬度
*/

void llToNes(double &longitude, double &latitude,double opoint_long = 113.006379,double opoint_lat = 28.230465);


/**
* 东北天的平面坐标转换为gps坐标
*
*@x  　　      目标x
*@y 　　　　　　     目标y
*@opoint_long  　　原点经度
*@opoint_lat   原点纬度
*/
void nesToLl(double &x, double &y,double opoint_long = 113.006379,double opoint_lat = 28.230465);


/**
* 东北天的平面坐标转换为:以矩形第４边为ｘ正轴的矩形
*
*@_points  　矩形的四个顶点的坐标，以逆时针方向，矩形的第三个点将成为原点，第４边将成为ｘ正轴
*@A 　　　　　　　 输出，转换矩阵
*@l_x  　　　　　输出，矩形的宽
*@l_y   　　输出，矩形的高
*/
void nesToBody(std::vector<std::vector<double> > &_points , Eigen::MatrixXd & A,double &l_x ,double &l_y);



/**
* 最小二乘拟合　ｙ　＝　ａｘ　＋　ｂ
*
*@_points  　　　存放顺序为：人的经纬度　正方形四个框的经纬度(逆时针)　边境的经纬度
*             每一个点是一个size为２的std::vector<double>,[0]号元素为经度，[1]号元素为纬度
*@a 　　　　　　　 　　输出，直线方程系数　"a"
*@b  　　　　　　　　　输出，直线方程系数　“ｂ”
*/
int leastSquare(const std::vector< std::vector<double> >& points,double &a ,double &b);

int leastSquare(const std::vector< std::vector<double> >& points,int size,double &a ,double &b);



}



#endif
