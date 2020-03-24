# topic: "/machine_learning/detection"
# date:  2019-07-02
string platform_id              #platform_id of different robot
time stamp                      #stamp means time attribute from origin data
string algorithm_id             #string_id of different algorithm
string[]   classes              #classes from one img
float32[]   scores              #scores for classes
geometry_msgs/Polygon[] bboxes  #box for each anchor
sensor_msgs/Image img_result    #a temporary solution

