#topic_name:/DataQueryResult
#date: 2018-05-24
string actor_name  #bind actor_name
string role_name   #sensor_name
string role_time   #data_time
string platform_id
geometry_msgs/Pose platform_pose
geometry_msgs/Pose sensor_pose
string data_type     #ROS message data type
uint8[] general_data #serialized data
