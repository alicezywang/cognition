坐标系为北东地 z坐标系要加负号
小车的话题/enemy/ground_truth/state 可通过rostopic echo /enemy/ground_truth/state查看具体数值
消息格式为nav_msgs/Odometry

  geometry_msgs/Pose pose
    geometry_msgs/Point position
      float64 x
      float64 y
      float64 z

  geometry_msgs/Twist twist
    geometry_msgs/Vector3 linear
      float64 x
      float64 y
      float64 z
    geometry_msgs/Vector3 angular
      float64 x
      float64 y
      float64 z

无人机的话题分别为/fixedwing_0/truth /fixedwing_1/truth /fixedwing_2/truth 可通过rostopic echo /fixedwing_0/truth查看具体数值
消息格式为rosplane_msgs/State

float32[3] position

float32 Vn x速度
float32 Ve y速度
float32 Vd z速度

小车被打击话题为/car_HP 
