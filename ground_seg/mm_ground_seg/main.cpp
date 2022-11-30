#include <ros/ros.h>
#include "mm_seg_node.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "mm_ground_seg");
    ros::NodeHandle node("~");
    mm::MMSegNode seg_node(node);
    ros::spin();
    return 0; 
}