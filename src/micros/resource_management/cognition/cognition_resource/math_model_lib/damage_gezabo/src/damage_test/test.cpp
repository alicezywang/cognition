#include <damage/damage.h>
#include <iostream>
#include <vector>

using namespace cooperative_damage_model;
using namespace std;

int main(int argc, char** argv)
{
  // Initialize the parameters 
  for(int exp=0;exp<=100;exp++){
    cout<<"Experiments: "<<exp;
    int robot_num=rand()%10;
    cout<<"   Robot_Number: "<<robot_num<<endl;
    float car_HP;
    robot_inf robot;
    multi_robot_inf multi_robot;
    float ori_c=(rand()%100)*(-0.1)+robot_num;
    float ori_r=(rand()%100)*(-0.1)+robot_num;
    for(int i=0; i<robot_num; i++){ 
      robot.position.x=-ori_r+i;
      robot.position.y=ori_r+i;
      robot.position.z=20+ori_r;
      robot.velocity_x=1.2*ori_r;
      robot.velocity_y=0.8*ori_r;
      robot.velocity_z=1.5*ori_r;
      robot.lethality=rand()%10+10;
      multi_robot.insert(multi_robot.begin()+i,robot);
    }
    car_inf car;
    car.position.x=ori_c;
    car.position.y=ori_c;
    car.position.z=0;
    car.velocity_x=1.2*ori_c; 
    car.velocity_y=1.5*ori_c;
    car_HP=cooperative_damage(multi_robot,car);
    if (car_HP==0){
      cout<<"The car is completely damaged!"<<endl;
    }
    else{
      if(car_HP==10){
        cout<<"The car is not damaged!"<<endl;
      }
      else{
        cout<<"The damage degree of the car: "<<car_HP<<endl;
      }
    }
  }
  return 0;
}
