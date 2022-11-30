#include <ros/ros.h>
#include "line_seg_node.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "line_gound_seg");
    ros::NodeHandle node("~");

    line::LineSegNode seg_node(node);
    ros::spin();
    return 0;
}