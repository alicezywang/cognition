#include "damage/damage.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>

namespace cooperative_damage_model
{
    float Distance(robot_inf robot,car_inf car){
        float x,y,z,v_x,v_y,v_z;
        float height,time_all,delta;
        float sandbag_x,sandbag_y,car_x,car_y,distance_car_sandbag;
        x=robot.position.x;  //the x-axis of robot
        y=robot.position.y;  //the y-axis of robot
        z=robot.position.z;
        v_x=robot.velocity_x;  //the angle-velocity of robot
        v_y=robot.velocity_y;    //the forward-velocity of robot
        v_z=robot.velocity_z;
        height=z;
        if (v_z==0){
            time_all=sqrt(2*height/g);  
        }
        else{
            delta=pow(v_z,2)+2*g*height;
            time_all=(sqrt(delta)+v_z)/g;
        }
        sandbag_x=robot.position.x+v_x*time_all;
        sandbag_y=robot.position.y+v_y*time_all;;
        car_x=car.position.x+time_all*car.velocity_x;  //the x-position of car
        car_y=car.position.y+time_all*car.velocity_y;  //the y-position of car
        distance_car_sandbag=sqrt(pow(fabs(car_x-sandbag_x),2)+pow(fabs(car_y-sandbag_y),2));//the distance between car and sandbag
        return distance_car_sandbag;
    }
    float cooperative_damage(multi_robot_inf multi_robot, car_inf car){
        int robot_num=multi_robot.size();
        float distance_car_sandbag[robot_num];
        float damage_capacity[robot_num];
        float car_HP;
        car_HP=10;  //the initial Health Point of car
        for(int i=1;i<=robot_num;i++){
            distance_car_sandbag[i]=Distance(multi_robot[i],car);
            damage_capacity[i]=multi_robot[i].lethality;
            if (distance_car_sandbag[i]<=damage_capacity[i]){
                car_HP=car_HP-(damage_capacity[i]/distance_car_sandbag[i]);
            }
            if (car_HP<0){
                car_HP=0;
            }
        }
        return car_HP;
    }
}

