#pragma once
#if !defined __AUDITORY_SENSATION_AREA__MODEL_H_
#define __AUDITORY_SENSATION_AREA__MODEL_H_

#include <iostream>


namespace warning_expel_model
{

 /**
*喷洒模型半径判断
*＠volume　预评估声源声音大小，分贝
*＠hearing_min　人的听力下限,分贝
*/
float auditorySensationArea(float source_volume,float hearing_min);



}
#endif
