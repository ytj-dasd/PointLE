#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <chrono>

#include "mm_seg_node.h"

int main(int argc, char** argv) {
    
    ros::init(argc, argv, "mm_ground_seg_test");
    ros::NodeHandle node("~");
    
    mm::MMSegNode seg_node(node);
    ROS_INFO("------TEST-----------");
    std::string cloud_path = "/home/zs/zs/master/experiment/ground_seg/special_scene/jiayi/2430.ply";
    std::string cloud_save = "/home/zs/zs/kitti/data/";
    ROS_INFO("cloud_path: %s", cloud_path.c_str());
    ROS_INFO("cloud_path: %s", cloud_save.c_str());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<PointXYZLO>::Ptr cloud_label(new pcl::PointCloud<PointXYZLO>);
    pcl::io::loadPLYFile(cloud_path, *cloud);
    auto start = std::chrono::steady_clock::now();
    seg_node.groundSeg(cloud, *cloud_label);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dur = end - start;
    ROS_INFO("run time: %f ms", dur.count());
    pcl::io::savePCDFile(cloud_save + "mm_ground_label.pcd", *cloud_label);
    ros::shutdown();
    return 0;
}