#topoic_name: /observer/getImages
#date:        2018-05-22
int32 img_type                 #1.new_img; 2.old_img
float32 publish_freq           #HZ
uint32 publish_num             #num
string begin_time              #20171024115723
string end_time                #old_img
sensor_msgs/NavSatFix[] points #two points:{northwest, southeast}
