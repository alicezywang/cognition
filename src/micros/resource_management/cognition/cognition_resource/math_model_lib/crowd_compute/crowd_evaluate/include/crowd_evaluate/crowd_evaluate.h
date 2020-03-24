#ifndef CROWD_EVALUATE_H
#define CROWD_EVALUATE_H
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

namespace warning_expel_model
{
  //#######################################
  // Define behaviors and indices
  //#######################################
  enum Behavior
  {
    CROWD_MOTIONLESS,
    CROWD_FORWARDS,
    CROWD_BACKWARDS,
    CROWD_PANIC
  };

  enum SequenceIndex
  {
    INDEX_LEFT = 0,
    INDEX_RIGHT = 1,
    INDEX_MAX = 2,
    INDEX_MIN = 3
  };

  enum EnergyIndex
  {
    INDEX_KINETIC = 0,
    INDEX_MUDDLE = 1,
    INDEX_FORWARD = 2,
    INDEX_DISPERSE = 3
  };

  #ifndef PI
  #define PI 3.1415926
  #endif

  #ifndef INF_VALUE
  #define INF_VALUE 1e20
  #endif

  #ifndef TINY_VALUE
  #define TINY_VALUE 1e-20
  #endif

  #ifndef K_ENERGY_THRES
  #define K_ENERGY_THRES 0.25
  #endif

  #ifndef M_ENERGY_THRES
  #define M_ENERGY_THRES 10
  #endif

  //#######################################
  // Define variables and structures
  //#######################################
  struct Point{
    float x;
    float y;
    float z;
  };

  // position of soldier
  struct SoldierPose
  {
    double x;	// position x
    double y;	// position y
    double x_velocity;	// velocity in x-direction
    double y_velocity;	// velocity in y-direction
  };

  // frame composed of several soldiers
  typedef std::vector<SoldierPose> SoldierArraySeq;

  // sequence of several frames
  struct BattleArray
  {
    SoldierArraySeq battle_array;
  };
  typedef std::vector<BattleArray> BattleArraySeq;

  // pure velocities in x-direction and y-direction
  struct Velocity
  {
    float x_velocity;
    float y_velocity;
  };

  // pure positions in x-direction and y-direction
  struct Coordinate
  {
    float x;
    float y;
  };

  // line of the frontier
  struct Line
  {
    Coordinate start_point;
    Coordinate end_point;
  };

  // four types of crowd energy
  struct Energy
  {
    float kinetic;
    float muddle;
    float forward;
    float disperse;
  };

  // four statistics of a sequence
  struct HalfSeq
  {
    float front_average;
    float end_average;
    float max_value;
    float min_value;
  };

  //#######################################
  // Define headers of functions
  //#######################################
  Behavior predictCrowdBehavior(const BattleArraySeq &, Line);	// predict behavior of a crowd
  Velocity calcuCrowdVelocity(const BattleArraySeq &);	// calculate summarized velocity of a crowd
  Coordinate calcuCrowdCenter(const BattleArraySeq &);	// calculate centralized coordinate of a crowd
  Coordinate calcuSceneCenter(const SoldierArraySeq &);	// calculate centralized coordinate of a scene
  Velocity calcuSceneVelocity(const SoldierArraySeq &);	// calculate summarized velocity of a scene
  Energy calcuSceneEnergy(const SoldierArraySeq &, Line);	// calculate energies of each scene

  float calcuAngleCosine(Velocity, Velocity);		// calculate consine angle between two velocities
  float calcuForwardAngle(SoldierPose, Line);		// calculate forward angle of each soldier
  HalfSeq statsHalfSequence(const float []);		        // calculate left and right parts of a sequence
}

#endif // CROWD_EVALUATE_H
