#include <crowd_evaluate/crowd_evaluate.h>

using namespace warning_expel_model;

int main(int argc, char** argv)
{
  // Initialize the scene sequence
  BattleArraySeq scene_seq;
  for(int i=0; i<10; i++)
  {
    SoldierArraySeq soldier_seq;
    for(int j=0; j<10; j++)
    {
      SoldierPose soldier;
      soldier.x = i+1;
      soldier.y = j+1;
      soldier.x_velocity = -i * 0.1;
      soldier.y_velocity = j * 0.1;
      printf("scene: %d, soldier: %d; possition, x: %f, y: %f; x velocity: %f, y velocity: %f.\n",i,j,soldier.x,soldier.y,soldier.x_velocity,soldier.y_velocity);
      soldier_seq.insert(soldier_seq.begin() + j, soldier);
    }
    BattleArray battle_scene;
    battle_scene.battle_array = soldier_seq;
    scene_seq.insert(scene_seq.begin() + i, battle_scene);
  }

  // Initialize the frontier
  Line frontier;
  frontier.start_point.x = 0;
  frontier.start_point.y = 0;
  frontier.end_point.x = 0;
  frontier.end_point.y = 10;
  int behavior = -1;
  behavior = predictCrowdBehavior(scene_seq, frontier);

  switch (behavior)
  {
    case CROWD_MOTIONLESS:
      printf("Crowd behavior: Motionless.\n");
      break;
    case CROWD_FORWARDS:
      printf("Crowd behavior: Forwards.\n");
      break;
    case CROWD_BACKWARDS:
      printf("Crowd behavior: Backwards.\n");
      break;
    case CROWD_PANIC:
      printf("Crowd behavior: Panic.\n");
      break;
    default:
      printf("Unrecognized behavior.\n");
      break;
  }

  return 0;
}
