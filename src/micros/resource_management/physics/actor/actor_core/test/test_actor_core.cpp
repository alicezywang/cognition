#include "actor_core/ACB.h"
#include "actor_core/actor_core.h"
#include "actor_core/actor_types.h"
#include "ros/ros.h"
#include <gtest/gtest.h>

TEST(ActorCore,initACBFromTask) {
	
}

int main(int argc,char **argv) {
	testing::InitGoogleTest(&argc, argv);  
	ros::init(argc, argv, "actor_core_test");
	return RUN_ALL_TESTS();
}
