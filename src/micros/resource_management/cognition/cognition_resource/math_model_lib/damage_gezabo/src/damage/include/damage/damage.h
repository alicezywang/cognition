#ifndef DAMAGE_H
#define DAMAGE_H
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>

namespace cooperative_damage_model{
  
  #ifndef g
  #define g 0.98
  #endif

  //#######################################
  // Define variables and structures
  //#######################################56
  struct Coordinate{
    float x;
    float y;
    float z;
  };

  struct robot_inf{
    Coordinate position;//无人机坐标
    float velocity_x; 
    float velocity_y; 
    float velocity_z; 
    float lethality; //打击毁伤能力[0,10]
  };
  typedef std::vector<robot_inf> multi_robot_inf;  //多个robot

  struct car_inf{
    Coordinate position;//无人车
    float velocity_x; //前进速度
    float velocity_y; //前进速度
  };

  //#######################################
  // Define headers of functions
  //#######################################
  float Distance(robot_inf,car_inf);
  float cooperative_damage(multi_robot_inf, car_inf);
}

#endif 
