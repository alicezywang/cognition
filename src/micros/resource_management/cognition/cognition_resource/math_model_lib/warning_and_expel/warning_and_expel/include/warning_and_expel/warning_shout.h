#pragma once
#if !defined __WARNING_SHOUT_MODEL__H_
#define __WARNING_SHOUT_MODEL__H_

#include <iostream>
#include <vector>
// using namespace std;
namespace warning_expel_model
{

	/**
  *声音模型半径判断
  *@h　无人机高度
  *＠angle 扬声器声音扩散角度
  *＠volume　扬声器声音大小，分贝
  *＠dB　预评估设置声音大学，分贝
   */

   float warningShout(float h,float dB,float volume = 120,float angle = 75);


}
#endif
