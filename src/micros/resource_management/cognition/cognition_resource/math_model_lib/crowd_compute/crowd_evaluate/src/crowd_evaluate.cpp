#include "crowd_evaluate/crowd_evaluate.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

namespace warning_expel_model
{
  //#######################################
  // Implementation of functions
  //#######################################

  // Predict the behavior of the crowd
  Behavior predictCrowdBehavior(const BattleArraySeq & scene_seq, Line frontier)
  {
    Behavior behavior = CROWD_MOTIONLESS; // behavior of the crowd

    // Statistic energies of all scenes
    int scene_number = scene_seq.size();
    float crowd_kinetic[scene_number];
    float crowd_muddle[scene_number];
    float crowd_forward[scene_number];
    float crowd_disperse[scene_number];
    for(int i=0; i<scene_number; i++)
    {
      SoldierArraySeq scene;
      scene = scene_seq[i].battle_array;
      // Calculate four kinds of energies
      Energy scene_energy;
      scene_energy = calcuSceneEnergy(scene, frontier);
      crowd_kinetic[i] = scene_energy.kinetic;
      crowd_muddle[i] = scene_energy.muddle;
      crowd_forward[i] = scene_energy.forward;
      crowd_disperse[i] = scene_energy.disperse;
    }

    // Predict motionless by kinetic energy
    HalfSeq stats_energy;
    float average_crowd_energy = 0;

    stats_energy= statsHalfSequence(crowd_kinetic);
    if(stats_energy.front_average > stats_energy.end_average)
      average_crowd_energy = stats_energy.end_average;            // using right-half average if kinetic-energy decreases
    else
      average_crowd_energy = stats_energy.max_value;

    if(average_crowd_energy < K_ENERGY_THRES)		        // predict motionless by low velocity
      behavior = CROWD_MOTIONLESS;
    else
    {
      stats_energy = statsHalfSequence(crowd_muddle);
      if(stats_energy.front_average > stats_energy.end_average)
        average_crowd_energy = stats_energy.end_average;
      else
        average_crowd_energy = stats_energy.max_value;

      if(average_crowd_energy < M_ENERGY_THRES)
      {
        // Predict forwards and retreat by forwards energy
        stats_energy = statsHalfSequence(crowd_forward);
        if(stats_energy.front_average > stats_energy.end_average)
          average_crowd_energy = stats_energy.end_average;        // using right-half average if forward-energy decreases
        else
          average_crowd_energy = stats_energy.max_value;

        if(average_crowd_energy < PI/2)	                        // predict backwards by low muddleless and low cross-angle 
          behavior = CROWD_BACKWARDS;
        else
          behavior = CROWD_FORWARDS;				// predict forwards by low muddleless and high cross-angle
      }
      else
      {
        behavior = CROWD_PANIC;					// predict panic by high velocity and high muddleless
      }
    }

    return behavior;
  }

  // Calculate the average velocity of the crowd
  Velocity calcuCrowdVelocity(const BattleArraySeq & scene_seq)
  {
    Velocity crowd_velocity;
    crowd_velocity.x_velocity = 0;
    crowd_velocity.y_velocity = 0;

    // Average the velocity of crowd across scenes and soldiers
    int scene_number = scene_seq.size();
    for(int i=0; i<scene_number; i++)
    {
      SoldierArraySeq scene;
      scene = scene_seq[i].battle_array;

      int soldier_number = scene.size();
      float scene_x_velocity = 0;
      float scene_y_velocity = 0;
      for(int j=0; j<soldier_number; j++)
      {
        SoldierPose soldier;
        soldier = scene[j];
        scene_x_velocity += soldier.x_velocity;
        scene_y_velocity += soldier.y_velocity;
      }
      crowd_velocity.x_velocity += scene_x_velocity / soldier_number;
      crowd_velocity.y_velocity += scene_y_velocity / soldier_number;
    }
    crowd_velocity.x_velocity = crowd_velocity.x_velocity / scene_number;
    crowd_velocity.y_velocity = crowd_velocity.y_velocity / scene_number;

    return crowd_velocity;
  }

  // Calculate the center of the crowd
  Coordinate calcuCrowdCenter(const BattleArraySeq & scene_seq)
  {
    Coordinate crowd_center;
    crowd_center.x = 0;
    crowd_center.y = 0;

    // Average the velocity of corwd across scenes and soldiers
    int scene_number = scene_seq.size();
    for(int i=0; i<scene_number; i++)
    {
      SoldierArraySeq scene;
      scene = scene_seq[i].battle_array;

      int soldier_number = scene.size();
      float scene_x_coordinate = 0;
      float scene_y_coordinate = 0;
      for (int j=0; j<soldier_number; j++)
      {
        SoldierPose soldier;
        soldier = scene[j];
        scene_x_coordinate += soldier.x;
        scene_y_coordinate += soldier.y;
      }
      crowd_center.x += scene_x_coordinate / soldier_number;
      crowd_center.y += scene_y_coordinate / soldier_number;
    }
    crowd_center.x = crowd_center.x / scene_number;
    crowd_center.y = crowd_center.y / scene_number;

    return crowd_center;
  }

  // Calculate the velocity of a scene
  Velocity calcuSceneVelocity(const SoldierArraySeq & scene)
  {
    Velocity scene_velocity;
    scene_velocity.x_velocity = 0;
    scene_velocity.y_velocity = 0;

    // Average the velocity of soldiers
    int soldier_number = scene.size();
    for(int j=0; j<soldier_number; j++)
    {
      SoldierPose soldier;
      soldier = scene[j];
      scene_velocity.x_velocity += soldier.x_velocity;
      scene_velocity.y_velocity += soldier.y_velocity;
    }
    scene_velocity.x_velocity = scene_velocity.x_velocity / soldier_number;
    scene_velocity.y_velocity = scene_velocity.y_velocity / soldier_number;

    return scene_velocity;
  }

  // Calculate the center of a scene
  Coordinate calcuSceneCenter(const SoldierArraySeq & scene)
  {
    Coordinate scene_center;
    scene_center.x = 0;
    scene_center.y = 0;

    // Calculate the center coordinate of all soldiers
    int soldier_number = scene.size();
    for(int j=0; j<soldier_number; j++)
    {
      SoldierPose soldier;
      soldier = scene[j];
      scene_center.x += soldier.x;
      scene_center.y += soldier.y;
    }
    scene_center.x = scene_center.x / soldier_number;
    scene_center.y = scene_center.y / soldier_number;
   
    return scene_center;
  }

  // Calculate four types of energies of a scene
  Energy calcuSceneEnergy(const SoldierArraySeq & scene, Line frontier)
  {
    int soldier_number=scene.size();

    // Calculate kinetic energy
    float kinetic_energy = 0;
    float delta = 1e-0;	// TODO: parameter over the kinetic energy
    for(int j=0; j<soldier_number; j++)
    {
      SoldierPose soldier;
      soldier = scene[j];
      kinetic_energy += pow(soldier.x_velocity, 2) + pow(soldier.y_velocity, 2);
    }
    kinetic_energy = delta * kinetic_energy / soldier_number;
	
    // Calculate muddleless energy
    float muddle_energy = 0;
    float alpha = 0.6;	// TODO: percentage of angle cosine in muddleless 
    float beta = 0.4;	// TODO: percentage of differene of velocities
    float gamma =	1e-0;	// TODO: parameter over the muddleless energy
    for(int i=0; i<soldier_number-1; i++)
    {
      for(int j=i+1; j<soldier_number; j++)
      {
        SoldierPose soldier1, soldier2;
        soldier1 = scene[i];
        soldier2 = scene[j];
        Velocity velocity1, velocity2;
        velocity1.x_velocity = soldier1.x_velocity;
        velocity1.y_velocity = soldier1.y_velocity;
        velocity2.x_velocity = soldier2.x_velocity;
        velocity2.y_velocity = soldier2.y_velocity;
        muddle_energy += alpha * calcuAngleCosine(velocity1, velocity2);
        float magnitude1 = sqrt(pow(velocity1.x_velocity, 2) + pow(velocity1.y_velocity, 2));
        float magnitude2 = sqrt(pow(velocity2.x_velocity, 2) + pow(velocity2.y_velocity, 2));
        muddle_energy += beta * (fabs(magnitude1 - magnitude2));
      }
    }
    int total_number = (soldier_number - 1) * (soldier_number - 2) / 2;
    muddle_energy = gamma * muddle_energy / total_number;

    // Calculate forward energy
    float forward_energy = 0;
    for(int j=0; j<soldier_number; j++)
    {
      SoldierPose soldier;
      soldier = scene[j];
      forward_energy += calcuForwardAngle(soldier, frontier);
    }
    forward_energy = forward_energy / soldier_number;

    // Calculate disperse energy
    float disperse_energy = 0;
    float sigma = 1e-0;	// TODO: parameter over disperse energy
    Coordinate scene_center = calcuSceneCenter(scene);
    for(int j=0; j<soldier_number; j++)
    {
      SoldierPose soldier;
      soldier = scene[j];
      disperse_energy += sqrt(pow(soldier.x - scene_center.x, 2) + pow(soldier.y - scene_center.y, 2));
    }
    disperse_energy = sigma * disperse_energy / soldier_number;

    // Assemble scene energies
    Energy scene_energy;
    scene_energy.kinetic = kinetic_energy;
    scene_energy.muddle = muddle_energy;
    scene_energy.forward = forward_energy;
    scene_energy.disperse = disperse_energy;
    
    return scene_energy;
  }

  // Calculate the angle between the velocity and the direction to the frontier
  float calcuForwardAngle(SoldierPose soldier, Line frontier)
  {
    Coordinate coordinate;
    coordinate.x = soldier.x;
    coordinate.y = soldier.y;
    Velocity velocity;
    velocity.x_velocity = soldier.x_velocity;
    velocity.y_velocity = soldier.y_velocity;
    // Calculate the foot point from the solider to the frontier
    Coordinate foot_point;
    float t1, t2, t;
    t1 = 0;
    t2 = 0;
    t = 0;
    t1 += (coordinate.x - frontier.start_point.x) * (frontier.start_point.x - frontier.end_point.x);
    t1 += (coordinate.y - frontier.start_point.y) * (frontier.start_point.y - frontier.end_point.y);
    t2 += pow(frontier.start_point.x - frontier.end_point.x, 2) + pow(frontier.start_point.y - frontier.end_point.y, 2);
    t = t1 / t2;
    foot_point.x = (frontier.start_point.x - frontier.end_point.x) * t + frontier.start_point.x;
    foot_point.y = (frontier.start_point.y - frontier.end_point.y) * t + frontier.start_point.y;
    Velocity velocity_twf;
    // Calculate the direction from the crowd center to the foot point
    velocity_twf.x_velocity = foot_point.x - coordinate.x;
    velocity_twf.y_velocity = foot_point.y - coordinate.y;
    // Calculte the angle of two velocities
    float angle_cosine = calcuAngleCosine(velocity_twf, velocity);
    float forward_angle = acos(angle_cosine);

    return forward_angle;
  }

  // Calculate the cosine value of the angle between two directions
  float calcuAngleCosine(Velocity velocity1, Velocity velocity2)
  {
    float angle_cosine = 0;
    angle_cosine = (velocity1.x_velocity * velocity2.x_velocity + velocity1.y_velocity * velocity2.y_velocity);
    angle_cosine /= std::max(sqrt(pow(velocity1.x_velocity, 2) + pow(velocity1.y_velocity, 2)), (double)TINY_VALUE);    // To avoid numerical problem
    angle_cosine /= std::max(sqrt(pow(velocity2.x_velocity, 2) + pow(velocity2.y_velocity, 2)), (double)TINY_VALUE);

    return angle_cosine;
  }

  // Statistic front and end averages and the max and min values of a sequence
  HalfSeq statsHalfSequence(const float sequence[])
  {
    float left_sum = 0;
    float right_sum = 0;
    int left_num = 0;
    int right_num = 0;
    float max_value = 0;
    float min_value = INF_VALUE;
    int scene_num = sizeof(sequence)/sizeof(sequence[0]);
    for(int i=0; i<scene_num; i++)
    {
      if(i < ceil(scene_num/2))	// front part
      {
        left_sum += sequence[i];
        left_num += 1;
      }
      else		        // end part
      {
        right_sum += sequence[i];
        right_num += 1;
      }
      if(sequence[i] > max_value) // maximum value
        max_value = sequence[i];
      if(sequence[i] < min_value)	// minimum value
        min_value = sequence[i];
    }

    HalfSeq stats_seq;
    stats_seq.front_average = left_sum / left_num;
    stats_seq.end_average = right_sum / right_num;
    stats_seq.max_value = max_value;
    stats_seq.min_value = min_value;

    return stats_seq;
  }
}
