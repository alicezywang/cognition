#include <math.h>
#include <warning_and_expel/auditory_sensation_area.h>
#define pi 3.1415926  

namespace warning_expel_model
{

float auditorySensationArea(float source_volume,float hearing_min){
  float k=source_volume-hearing_min;
  float r=pow(10,(k-10*log10(4*pi))/20);
  return r;
}

}
