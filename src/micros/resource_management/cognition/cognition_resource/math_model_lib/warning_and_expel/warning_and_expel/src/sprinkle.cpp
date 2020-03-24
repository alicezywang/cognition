#include <warning_and_expel/sprinkle.h>
#include <warning_and_expel/model_util.h>
#include <math.h>
#include <vector>
#define pi 3.1415926

// using namespace std;
namespace warning_expel_model
{
	
float sprinkle(float h,double angle)
{
  //angle为喷洒器的喷洒角度
  float radian=angle*pi/180;
  float r=h*tan(radian);
  return r;
}


int sprinkleLineInterval(float interval,float h,double opoint_long,double opoint_lat,
                           int amount,int leader,std::vector<double> & lons,float & r,double angle)

{
  double x , y;
  amount--;
  y = 0;
  r = sprinkle(h,angle);
  lons.clear();
  if(leader < 0 || leader > amount){
    return -1;
  }
  for(int i=0 ; i<= amount ; i++){
    y = 0;
    if(i == leader){
      lons.push_back(opoint_long);
      continue;
    }
    x = (i-leader)*interval;
    nesToLl(x, y,opoint_long ,opoint_lat);
    lons.push_back(x);
  }
  return 0;

}


int sprinkleLineIntervalX(float interval,float h,double opoint_long,double opoint_lat,
                           int amount,int leader,std::vector<double> & lats,float & r,double angle)

{
  double x , y;
  amount--;
  x = 0;
  r = sprinkle(h,angle);
  lats.clear();
  if(leader < 0 || leader > amount){
    return -1;
  }
  for(int i=0 ; i<= amount ; i++){
    x = 0;
    if(i == leader){
      lats.push_back(opoint_lat);
      continue;
    }
    y = (i-leader)*interval;
    nesToLl(x, y,opoint_long ,opoint_lat);
    lats.push_back(y);
  }
  return 0;

}



int sprinkleLineInterval(float interval,float h,
                           std::vector<std::vector<double> > &inflection_points,
                           int amount,int leader,
                           std::vector<std::vector<double> > & airplane_points,
                           float & r,bool & y_direction,double angle){

  if(inflection_points.size()<2){
    return -1;
  }
  if(abs(inflection_points[0][0]-inflection_points[1][0]) < abs(inflection_points[0][1] - inflection_points[1][1])){
    y_direction = true;
  }else{
    y_direction = false;
  }
  airplane_points.clear();
  std::vector<double> l_a;
  std::vector<double> l_b;
  std::vector<double> temp;
  for(int i=0 ; i<(inflection_points.size()/2); i++){
    l_a.clear();
    l_b.clear();
    temp.clear();
    if(y_direction){
       //同纬度，ｙ方向飞
       sprinkleLineInterval(interval,h,inflection_points[i*2][0],inflection_points[i*2][1],amount,leader,l_a,r,angle);
       sprinkleLineInterval(interval,h,inflection_points[i*2 + 1][0],inflection_points[i*2 +1][1],amount,leader,l_b,r,angle);
       for(int j=0 ; j< l_a.size() ; j++){
         temp.push_back(l_a[j]);
         temp.push_back(inflection_points[i*2][1]);
         temp.push_back(l_b[j]);
         temp.push_back(inflection_points[i*2+1][1]);
       }

    }else{
      //同经度，ｘ方向飞
       sprinkleLineIntervalX(interval,h,inflection_points[i*2][0],inflection_points[i*2][1],amount,leader,l_a,r,angle);
       sprinkleLineIntervalX(interval,h,inflection_points[i*2 + 1][0],inflection_points[i*2 +1][1],amount,leader,l_b,r,angle);
       for(int j=0 ; j< l_a.size() ; j++){
         temp.push_back(inflection_points[i*2][0]);
         temp.push_back(l_a[j]);
         temp.push_back(inflection_points[i*2+1][0]);
         temp.push_back(l_b[j]);
       }

    }
    airplane_points.push_back(temp);
     

  }
  return 0;
  
  

}

}
