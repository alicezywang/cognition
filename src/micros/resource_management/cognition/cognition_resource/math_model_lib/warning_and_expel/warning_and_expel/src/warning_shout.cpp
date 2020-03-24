#include <warning_and_expel/warning_shout.h>
#include <math.h>
#include <vector>
using namespace std;
#define pi 3.1415926  


namespace warning_expel_model
{

float warningShout(float h,float dB,float volume,float angle){
  float r;
  float radian=angle*pi/180/2;
  float k=volume-dB;
  float n=pow(10,(k-10*log10(4*pi))/20);//根据Lp=Lw-K,K=10log(10,4π)+20log(10,r),r为两点距离计算衰减
  float r1,r2;
  if(n*n < h*h){
    r1 = 0;
  }else{
    r1=sqrt(n*n-h*h);
  }

  r2=h*tan(radian);
  if(r1<=r2)
    r=r1;
  else
    r=r2;
//  std::cout<<"r1:"<<r1<<"  r2:"<<r2<<"  r："<<r<<std::endl;
  return r;


}


}
